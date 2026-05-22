/**
 * camera_stream.ino
 * ─────────────────────────────────────────────────────────────────────────────
 * ESP32-CAM Phase 1 Firmware
 *
 * Responsibilities:
 *   1. Detect SD card at boot → if present, record every session as a
 *      timestamped .mjpeg file.  If absent, stream-only mode (no crash).
 *   2. Serve an MJPEG HTTP stream on port 80 at /stream — always.
 *   3. Serve a /status JSON endpoint for Phase 2+ monitoring.
 *
 * Hardware target : AI-Thinker ESP32-CAM (OV2640)
 * Board package   : esp32 by Espressif ≥ 2.0.14
 * Libraries       : None beyond the ESP-IDF built-ins
 *
 * Pin map (AI-Thinker default — do NOT change):
 *   PWDN=32  RESET=-1  XCLK=0   SIOD=26  SIOC=27
 *   Y9=35    Y8=34     Y7=39    Y6=36    Y5=21
 *   Y4=19    Y3=18     Y2=5     VSYNC=25 HREF=23 PCLK=22
 *   SD_MMC: CLK=14, CMD=15, DATA0=2  (1-bit mode)
 *   Flash LED: GPIO 4
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <Arduino.h>
#include <WiFi.h>
#include <SD_MMC.h>
#include <esp_camera.h>
#include <esp_http_server.h>

// ── Wi-Fi credentials — keep in secrets.h, excluded from version control ──────
// Create secrets.h alongside this file and define:
//   #define WIFI_SSID     "your-ssid"
//   #define WIFI_PASSWORD "your-password"
#define WIFI_SSID     "SSID"
#define WIFI_PASSWORD "PASSWORD"

// ── Camera pin map (AI-Thinker) ───────────────────────────────────────────────
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22
#define FLASH_LED_GPIO   4

// ── Streaming config ──────────────────────────────────────────────────────────
#define STREAM_PORT   80
#define TARGET_FPS    30

// ── MJPEG boundary constants — lengths precomputed to avoid per-frame strlen ──
static const char STREAM_CONTENT_TYPE[] =
    "multipart/x-mixed-replace;boundary=frame";
static const char STREAM_SEPARATOR[] =
    "\r\n--frame\r\n";
static const char STREAM_HEADER_FMT[] =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

static const size_t SEPARATOR_LEN = sizeof(STREAM_SEPARATOR) - 1;

// ── Globals ───────────────────────────────────────────────────────────────────
static bool           sdAvailable  = false;
static File           videoFile;
static int            streamClients = 0;
static portMUX_TYPE   clientMux     = portMUX_INITIALIZER_UNLOCKED;
static httpd_handle_t server        = NULL;

// Thread-safe counter helpers (FreeRTOS spinlock, safe on dual-core ESP32)
static inline void clientInc() { portENTER_CRITICAL(&clientMux); streamClients++; portEXIT_CRITICAL(&clientMux); }
static inline void clientDec() { portENTER_CRITICAL(&clientMux); streamClients--; portEXIT_CRITICAL(&clientMux); }
static inline int  clientGet() { int n; portENTER_CRITICAL(&clientMux); n = streamClients; portEXIT_CRITICAL(&clientMux); return n; }

// ─────────────────────────────────────────────────────────────────────────────
// Camera initialisation
// ─────────────────────────────────────────────────────────────────────────────
static bool initCamera() {
    camera_config_t cfg = {};
    cfg.ledc_channel = LEDC_CHANNEL_0;
    cfg.ledc_timer   = LEDC_TIMER_0;
    cfg.pin_d0       = Y2_GPIO_NUM;
    cfg.pin_d1       = Y3_GPIO_NUM;
    cfg.pin_d2       = Y4_GPIO_NUM;
    cfg.pin_d3       = Y5_GPIO_NUM;
    cfg.pin_d4       = Y6_GPIO_NUM;
    cfg.pin_d5       = Y7_GPIO_NUM;
    cfg.pin_d6       = Y8_GPIO_NUM;
    cfg.pin_d7       = Y9_GPIO_NUM;
    cfg.pin_xclk     = XCLK_GPIO_NUM;
    cfg.pin_pclk     = PCLK_GPIO_NUM;
    cfg.pin_vsync    = VSYNC_GPIO_NUM;
    cfg.pin_href     = HREF_GPIO_NUM;
    cfg.pin_sscb_sda = SIOD_GPIO_NUM;
    cfg.pin_sscb_scl = SIOC_GPIO_NUM;
    cfg.pin_pwdn     = PWDN_GPIO_NUM;
    cfg.pin_reset    = RESET_GPIO_NUM;
    cfg.xclk_freq_hz = 20000000;
    cfg.pixel_format = PIXFORMAT_JPEG;
    cfg.frame_size   = FRAMESIZE_QVGA;  // 320×240 — good balance at 30 fps
    cfg.jpeg_quality = 12;              // 0=best, 63=worst; 10–15 is practical
    cfg.fb_count     = 2;               // double-buffer
    cfg.grab_mode    = CAMERA_GRAB_LATEST;

    esp_err_t err = esp_camera_init(&cfg);
    if (err != ESP_OK) {
        Serial.printf("[CAM] Init failed: 0x%x\n", err);
        return false;
    }
    Serial.printf("[CAM] Initialised — QVGA JPEG @ %d fps\n", TARGET_FPS);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// SD card initialisation (optional — stream works without SD)
// ─────────────────────────────────────────────────────────────────────────────
static bool initSD() {
    // 1-bit MMC avoids GPIO 13 (flash LED conflict on some board revisions)
    if (!SD_MMC.begin("/sdcard", /*mode1bit=*/true)) {
        Serial.printf("[SD] Mount failed — recording disabled\n");
        return false;
    }
    if (SD_MMC.cardType() == CARD_NONE) {
        Serial.printf("[SD] No card inserted — recording disabled\n");
        SD_MMC.end();
        return false;
    }
    Serial.printf("[SD] Card ready — %.0f MB free\n",
                  (float)SD_MMC.totalBytes() / (1024.0f * 1024.0f));
    return true;
}

// Opens a .mjpeg file named by uptime-ms. Called once after SD is confirmed.
static bool openVideoFile() {
    char path[40];
    snprintf(path, sizeof(path), "/session_%lu.mjpeg", millis());
    videoFile = SD_MMC.open(path, FILE_WRITE);
    if (!videoFile) {
        Serial.printf("[SD] Could not create %s\n", path);
        return false;
    }
    Serial.printf("[SD] Recording → %s\n", path);
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// MJPEG HTTP stream handler
// ─────────────────────────────────────────────────────────────────────────────
static esp_err_t handleStream(httpd_req_t* req) {
    clientInc();
    Serial.printf("[STREAM] Client connected — total: %d\n", clientGet());

    httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

    char     hdr[64];
    int      writeCount = 0;   // per-connection counter, not shared static
    esp_err_t res       = ESP_OK;

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(5));
            continue;
        }

        // Record to SD if available and file is valid
        if (sdAvailable && videoFile) {
            videoFile.write(fb->buf, fb->len);
            if (++writeCount % 30 == 0) videoFile.flush();
        }

        // Boundary
        res = httpd_resp_send_chunk(req, STREAM_SEPARATOR, SEPARATOR_LEN);
        if (res != ESP_OK) { esp_camera_fb_return(fb); break; }

        // Per-frame header
        size_t hlen = snprintf(hdr, sizeof(hdr), STREAM_HEADER_FMT, fb->len);
        res = httpd_resp_send_chunk(req, hdr, hlen);
        if (res != ESP_OK) { esp_camera_fb_return(fb); break; }

        // Frame data — return buffer immediately after sending
        res = httpd_resp_send_chunk(req, (const char*)fb->buf, fb->len);
        esp_camera_fb_return(fb);
        if (res != ESP_OK) break;

        vTaskDelay(pdMS_TO_TICKS(1));
    }

    clientDec();
    Serial.printf("[STREAM] Client disconnected — total: %d\n", clientGet());
    return ESP_OK;
}

// ── /status JSON endpoint (stub for Phase 2+ monitoring) ─────────────────────
static esp_err_t handleStatus(httpd_req_t* req) {
    char json[128];
    snprintf(json, sizeof(json),
             "{\"sd_present\":%s,\"uptime_ms\":%lu,\"clients\":%d,\"fps\":%d}",
             sdAvailable ? "true" : "false",
             millis(),
             clientGet(),
             TARGET_FPS);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_sendstr(req, json);
    return ESP_OK;
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP server setup
// ─────────────────────────────────────────────────────────────────────────────
static void startWebServer() {
    httpd_config_t cfg  = HTTPD_DEFAULT_CONFIG();
    cfg.server_port     = STREAM_PORT;
    cfg.max_uri_handlers = 4;

    if (httpd_start(&server, &cfg) != ESP_OK) {
        Serial.printf("[HTTP] Failed to start server\n");
        return;
    }

    // Register /stream
    httpd_uri_t uri_stream = {};
    uri_stream.uri     = "/stream";
    uri_stream.method  = HTTP_GET;
    uri_stream.handler = handleStream;
    httpd_register_uri_handler(server, &uri_stream);

    // Register /status
    httpd_uri_t uri_status = {};
    uri_status.uri     = "/status";
    uri_status.method  = HTTP_GET;
    uri_status.handler = handleStatus;
    httpd_register_uri_handler(server, &uri_status);

    Serial.printf("[HTTP] Server started on port %d\n", STREAM_PORT);
}

// ─────────────────────────────────────────────────────────────────────────────
// setup()
// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.printf("\n[BOOT] ESP32-CAM Phase 1 starting...\n");

    // Disable flash LED — strobes during SD access otherwise
    pinMode(FLASH_LED_GPIO, OUTPUT);
    digitalWrite(FLASH_LED_GPIO, LOW);

    // Camera (fatal if absent)
    if (!initCamera()) {
        Serial.printf("[BOOT] Camera init failed — restarting in 5 s\n");
        delay(5000);
        esp_restart();
    }

    // SD card (optional)
    sdAvailable = initSD();
    if (sdAvailable && !openVideoFile()) {
        sdAvailable = false;
    }

    // Wi-Fi
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
    Serial.printf("[WIFI] Connecting");
    for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) {
        delay(500);
        Serial.printf(".");
    }
    if (WiFi.status() != WL_CONNECTED) {
        Serial.printf("\n[WIFI] Failed — restarting in 5 s\n");
        delay(5000);
        esp_restart();
    }
    Serial.printf("\n[WIFI] Connected — IP: %s\n",
                  WiFi.localIP().toString().c_str());
    Serial.printf("[STREAM] http://%s/stream\n",
                  WiFi.localIP().toString().c_str());

    startWebServer();

    Serial.printf("[BOOT] Free heap: %u bytes — ready\n", ESP.getFreeHeap());
}

// ─────────────────────────────────────────────────────────────────────────────
// loop() — heartbeat only; all real work is in HTTP handler tasks
// ─────────────────────────────────────────────────────────────────────────────
void loop() {
    static unsigned long lastHB = 0;
    unsigned long now = millis();
    if (now - lastHB >= 10000) {
        lastHB = now;
        Serial.printf("[HB] uptime=%lu ms  sd=%s  clients=%d  heap=%u\n",
                      now,
                      sdAvailable ? "yes" : "no",
                      clientGet(),
                      ESP.getFreeHeap());
    }
    delay(100);
}