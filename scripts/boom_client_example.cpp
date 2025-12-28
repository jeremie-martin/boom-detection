/**
 * =============================================================================
 * BOOM DETECTION C++ CLIENT
 * =============================================================================
 *
 * Self-contained client for the boom detection Python server.
 * Communicates via Unix socket with the boom_server.py process.
 *
 * COMPILE:
 *   g++ -std=c++17 -O2 boom_client_example.cpp -o boom_client
 *
 * USAGE:
 *   # 1. Start Python server (keep running in background):
 *   python scripts/boom_server.py models/boom_v1 --socket /tmp/boom.sock
 *
 *   # 2. Run client:
 *   ./boom_client /tmp/boom.sock path /path/to/simulation_data.bin
 *   ./boom_client /tmp/boom.sock binary   # (uses generated test data)
 *
 * =============================================================================
 * PROTOCOL
 * =============================================================================
 *
 * Request format:
 *   [4 bytes: header_length (uint32 LE)]
 *   [header_length bytes: JSON header]
 *   [optional: binary data for "binary" mode]
 *
 * Response format:
 *   [4 bytes: response_length (uint32 LE)]
 *   [response_length bytes: JSON response]
 *
 * =============================================================================
 * RESPONSE EXAMPLES
 * =============================================================================
 *
 * CASE 1: BOOM DETECTED (accepted=true)
 * -------------------------------------
 * When the model confidently detects the boom frame:
 *
 *   {
 *     "status": "ok",
 *     "accepted": true,
 *     "boom_frame": 546,          <-- THE DETECTED BOOM FRAME
 *     "cnn_pred": 546,            <-- CNN model prediction
 *     "hgb_pred": 546,            <-- HistGBM model prediction
 *     "disagreement": 0,          <-- |cnn_pred - hgb_pred|
 *     "predicted_quality": 0.66,  <-- Quality score (0-1)
 *     "accept_score": 0.796       <-- Combined confidence (0-1)
 *   }
 *
 *   Interpretation:
 *   - boom_frame is the frame where the boom occurs
 *   - Both models agree (disagreement=0), high quality detected
 *   - Use this simulation for your animation
 *
 * CASE 2: BOOM NOT DETECTED / REJECTED (accepted=false)
 * ------------------------------------------------------
 * When the model rejects the simulation (low quality or model disagreement):
 *
 *   {
 *     "status": "ok",
 *     "accepted": false,
 *     "boom_frame": null,         <-- NULL when rejected!
 *     "cnn_pred": 367,            <-- CNN still made a prediction
 *     "hgb_pred": 412,            <-- HGB made a different prediction
 *     "disagreement": 45,         <-- High disagreement = uncertain
 *     "predicted_quality": 0.17,  <-- Low quality simulation
 *     "accept_score": 0.342       <-- Below threshold (0.60)
 *   }
 *
 *   Interpretation:
 *   - boom_frame is NULL - do not use this value!
 *   - The simulation has low quality or ambiguous boom
 *   - Discard this simulation and generate a new one
 *
 * CASE 3: ERROR
 * -------------
 * When something goes wrong (file not found, invalid data, etc.):
 *
 *   {
 *     "status": "error",
 *     "message": "File not found: /bad/path.bin"
 *   }
 *
 * =============================================================================
 * DATA FORMAT FOR BINARY MODE
 * =============================================================================
 *
 * When sending simulation data directly (not from file):
 *
 * - Shape: [frames][pendulums][8] as contiguous float32 array
 * - 8 values per pendulum per frame:
 *     [0] x1  - first joint x position
 *     [1] y1  - first joint y position
 *     [2] x2  - second joint (tip) x position
 *     [3] y2  - second joint (tip) y position
 *     [4] th1 - first segment angle
 *     [5] th2 - second segment angle
 *     [6] w1  - first segment angular velocity
 *     [7] w2  - second segment angular velocity
 *
 * - Expected: 2000 pendulums (model trained on this)
 * - Typical frame count: 500-2000 frames
 *
 * Memory layout example (2 frames, 3 pendulums):
 *   frame0_pend0: [x1,y1,x2,y2,th1,th2,w1,w2]
 *   frame0_pend1: [x1,y1,x2,y2,th1,th2,w1,w2]
 *   frame0_pend2: [x1,y1,x2,y2,th1,th2,w1,w2]
 *   frame1_pend0: [x1,y1,x2,y2,th1,th2,w1,w2]
 *   frame1_pend1: [x1,y1,x2,y2,th1,th2,w1,w2]
 *   frame1_pend2: [x1,y1,x2,y2,th1,th2,w1,w2]
 *
 * =============================================================================
 */

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// BOOM RESULT STRUCTURE
// =============================================================================

/**
 * Result from boom detection server.
 *
 * Check `accepted` first:
 * - If true:  use `boom_frame` for your animation
 * - If false: discard this simulation, boom_frame is not valid
 */
struct BoomResult {
    bool ok = false;           // true if request succeeded (check accepted next)
    bool accepted = false;     // true if boom was confidently detected

    // Only valid when accepted=true
    int boom_frame = -1;       // Frame where boom occurs (-1 if rejected)

    // Model predictions (always available when ok=true)
    int cnn_pred = -1;         // CNN model prediction
    int hgb_pred = -1;         // HistGBM model prediction
    int disagreement = -1;     // |cnn_pred - hgb_pred|

    // Confidence scores (always available when ok=true)
    float predicted_quality = 0.0f;  // Quality score (0-1)
    float accept_score = 0.0f;       // Combined confidence (0-1), threshold is 0.60

    // Error info (only when ok=false)
    std::string error_message;

    // Raw JSON response for debugging
    std::string raw_json;
};

// =============================================================================
// SIMPLE JSON PARSER (no dependencies)
// =============================================================================

namespace json {

// Extract string value: "key": "value" or "key": null
inline std::optional<std::string> get_string(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return std::nullopt;

    pos += pattern.length();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (pos >= json.size()) return std::nullopt;

    if (json[pos] == 'n') {  // null
        return std::nullopt;
    }

    if (json[pos] != '"') return std::nullopt;
    pos++;

    size_t end = json.find('"', pos);
    if (end == std::string::npos) return std::nullopt;

    return json.substr(pos, end - pos);
}

// Extract integer value: "key": 123 or "key": null
inline std::optional<int> get_int(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return std::nullopt;

    pos += pattern.length();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (pos >= json.size()) return std::nullopt;
    if (json[pos] == 'n') return std::nullopt;  // null

    bool negative = false;
    if (json[pos] == '-') {
        negative = true;
        pos++;
    }

    int value = 0;
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
        value = value * 10 + (json[pos] - '0');
        pos++;
    }

    return negative ? -value : value;
}

// Extract float value: "key": 0.123
inline std::optional<float> get_float(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return std::nullopt;

    pos += pattern.length();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (pos >= json.size()) return std::nullopt;
    if (json[pos] == 'n') return std::nullopt;  // null

    size_t end = pos;
    while (end < json.size() && (json[end] == '-' || json[end] == '.' ||
           (json[end] >= '0' && json[end] <= '9'))) {
        end++;
    }

    try {
        return std::stof(json.substr(pos, end - pos));
    } catch (...) {
        return std::nullopt;
    }
}

// Extract boolean value: "key": true/false
inline std::optional<bool> get_bool(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return std::nullopt;

    pos += pattern.length();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;

    if (json.compare(pos, 4, "true") == 0) return true;
    if (json.compare(pos, 5, "false") == 0) return false;
    return std::nullopt;
}

}  // namespace json

// =============================================================================
// BOOM CLIENT CLASS
// =============================================================================

/**
 * Client for communicating with boom_server.py.
 *
 * Usage:
 *   BoomClient client("/tmp/boom.sock");
 *   BoomResult result = client.predict_path("/path/to/sim.bin");
 *   if (result.ok && result.accepted) {
 *       std::cout << "Boom at frame " << result.boom_frame << "\n";
 *   }
 */
class BoomClient {
public:
    /**
     * Connect to the boom detection server.
     *
     * @param socket_path Path to Unix socket (e.g., "/tmp/boom.sock")
     * @throws std::runtime_error if connection fails
     */
    explicit BoomClient(const std::string& socket_path) {
        sock_ = socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock_ < 0) {
            throw std::runtime_error("Failed to create socket");
        }

        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

        if (connect(sock_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            close(sock_);
            throw std::runtime_error("Failed to connect to server at " + socket_path +
                                   ". Is boom_server.py running?");
        }
    }

    ~BoomClient() {
        if (sock_ >= 0) {
            close(sock_);
        }
    }

    // Non-copyable
    BoomClient(const BoomClient&) = delete;
    BoomClient& operator=(const BoomClient&) = delete;

    /**
     * Predict boom frame from a simulation file.
     *
     * The server loads the file and extracts features.
     * Use this when simulation data is already saved to disk.
     *
     * @param path Path to simulation_data.bin file
     * @return BoomResult with detection results
     */
    BoomResult predict_path(const std::string& path) {
        std::string header = R"({"type":"path","path":")" + path + R"("})";
        return parse_response(send_request(header, nullptr, 0));
    }

    /**
     * Predict boom frame from in-memory simulation data.
     *
     * Use this when you have simulation data in memory and don't
     * want to write it to disk first.
     *
     * @param data    Pointer to float32 array, shape [frames][pendulums][8]
     * @param frames  Number of frames in simulation
     * @param pendulums Number of pendulums (should be 2000 for best results)
     * @param values  Values per pendulum per frame (default: 8)
     * @return BoomResult with detection results
     *
     * Data layout: contiguous float32 array in row-major order
     *   data[f * pendulums * 8 + p * 8 + v] = value at frame f, pendulum p, value v
     */
    BoomResult predict_binary(const float* data, int frames, int pendulums, int values = 8) {
        std::string header = R"({"type":"binary","frames":)" + std::to_string(frames) +
                            R"(,"pendulums":)" + std::to_string(pendulums) +
                            R"(,"values":)" + std::to_string(values) + "}";

        size_t data_size = static_cast<size_t>(frames) * pendulums * values * sizeof(float);
        return parse_response(send_request(header, data, data_size));
    }

    /**
     * Request server shutdown.
     * Server will stop accepting new connections after this.
     */
    void shutdown() {
        std::string header = R"({"type":"shutdown"})";
        send_request(header, nullptr, 0);
    }

private:
    int sock_ = -1;

    std::string send_request(const std::string& header, const void* data, size_t data_size) {
        // Send header length (4 bytes, little-endian uint32)
        uint32_t header_len = static_cast<uint32_t>(header.size());
        send_all(&header_len, sizeof(header_len));

        // Send JSON header
        send_all(header.data(), header.size());

        // Send binary data if present
        if (data && data_size > 0) {
            send_all(data, data_size);
        }

        // Receive response
        return recv_response();
    }

    void send_all(const void* data, size_t size) {
        const char* ptr = static_cast<const char*>(data);
        size_t sent = 0;
        while (sent < size) {
            ssize_t n = send(sock_, ptr + sent, size - sent, 0);
            if (n <= 0) {
                throw std::runtime_error("Send failed - connection lost");
            }
            sent += static_cast<size_t>(n);
        }
    }

    void recv_all(void* data, size_t size) {
        char* ptr = static_cast<char*>(data);
        size_t received = 0;
        while (received < size) {
            ssize_t n = recv(sock_, ptr + received, size - received, 0);
            if (n <= 0) {
                throw std::runtime_error("Recv failed - connection lost");
            }
            received += static_cast<size_t>(n);
        }
    }

    std::string recv_response() {
        // Read response length (4 bytes, little-endian uint32)
        uint32_t resp_len;
        recv_all(&resp_len, sizeof(resp_len));

        // Read response JSON
        std::string response(resp_len, '\0');
        recv_all(response.data(), resp_len);

        return response;
    }

    BoomResult parse_response(const std::string& response) {
        BoomResult result;
        result.raw_json = response;

        // Check status
        auto status = json::get_string(response, "status");
        if (!status || *status != "ok") {
            result.ok = false;
            auto msg = json::get_string(response, "message");
            result.error_message = msg.value_or("Unknown error");
            return result;
        }

        result.ok = true;
        result.accepted = json::get_bool(response, "accepted").value_or(false);

        // boom_frame is null when rejected
        auto boom = json::get_int(response, "boom_frame");
        result.boom_frame = boom.value_or(-1);

        result.cnn_pred = json::get_int(response, "cnn_pred").value_or(-1);
        result.hgb_pred = json::get_int(response, "hgb_pred").value_or(-1);
        result.disagreement = json::get_int(response, "disagreement").value_or(-1);
        result.predicted_quality = json::get_float(response, "predicted_quality").value_or(0.0f);
        result.accept_score = json::get_float(response, "accept_score").value_or(0.0f);

        return result;
    }
};

// =============================================================================
// OPTIONAL: DOWNSAMPLING UTILITY
// =============================================================================
//
// If your simulation has more than 2000 pendulums, you should downsample
// before sending to the server. The model was trained on 2000 pendulums.
//
// NOTE: For best results, run your simulation with exactly 2000 pendulums
// instead of downsampling. This avoids any sampling bias.
//
// If you must downsample, use seed=42 to match Python's behavior.

/**
 * Downsample simulation from many pendulums to fewer.
 *
 * @param src Source data [frames][src_pendulums][8] as float32
 * @param frames Number of frames
 * @param src_pendulums Number of pendulums in source
 * @param dst Destination buffer [frames][dst_pendulums][8]
 * @param dst_pendulums Target count (default: 2000)
 * @param seed Random seed (default: 42 to match Python)
 */
inline void downsample_pendulums(
    const float* src,
    int frames,
    int src_pendulums,
    float* dst,
    int dst_pendulums = 2000,
    unsigned int seed = 42
) {
    if (src_pendulums <= dst_pendulums) {
        // No downsampling needed
        std::memcpy(dst, src,
            static_cast<size_t>(frames) * src_pendulums * 8 * sizeof(float));
        return;
    }

    // Select random pendulum indices
    std::mt19937 rng(seed);
    std::vector<int> indices(src_pendulums);
    for (int i = 0; i < src_pendulums; ++i) {
        indices[i] = i;
    }

    // Fisher-Yates partial shuffle
    for (int i = 0; i < dst_pendulums; ++i) {
        std::uniform_int_distribution<int> dist(i, src_pendulums - 1);
        std::swap(indices[i], indices[dist(rng)]);
    }

    // Sort for cache-friendly access
    std::sort(indices.begin(), indices.begin() + dst_pendulums);

    // Copy selected pendulums
    for (int f = 0; f < frames; ++f) {
        for (int p = 0; p < dst_pendulums; ++p) {
            const float* src_ptr = src + (static_cast<size_t>(f) * src_pendulums + indices[p]) * 8;
            float* dst_ptr = dst + (static_cast<size_t>(f) * dst_pendulums + p) * 8;
            std::memcpy(dst_ptr, src_ptr, 8 * sizeof(float));
        }
    }
}

// =============================================================================
// EXAMPLE MAIN
// =============================================================================

void print_result(const BoomResult& result) {
    std::cout << "\n";
    std::cout << "=== BOOM DETECTION RESULT ===\n";

    if (!result.ok) {
        std::cout << "Status: ERROR\n";
        std::cout << "Message: " << result.error_message << "\n";
        return;
    }

    std::cout << "Status: OK\n";
    std::cout << "Accepted: " << (result.accepted ? "YES" : "NO") << "\n";

    if (result.accepted) {
        std::cout << "\n*** BOOM DETECTED AT FRAME " << result.boom_frame << " ***\n";
    } else {
        std::cout << "\n*** SIMULATION REJECTED - generate a new one ***\n";
    }

    std::cout << "\nModel predictions:\n";
    std::cout << "  CNN prediction: " << result.cnn_pred << "\n";
    std::cout << "  HGB prediction: " << result.hgb_pred << "\n";
    std::cout << "  Disagreement:   " << result.disagreement << " frames\n";

    std::cout << "\nConfidence scores:\n";
    std::cout << "  Quality score:  " << result.predicted_quality << " (0-1)\n";
    std::cout << "  Accept score:   " << result.accept_score << " (threshold: 0.60)\n";

    std::cout << "\nRaw JSON:\n  " << result.raw_json << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Boom Detection C++ Client\n\n";
        std::cerr << "Usage: " << argv[0] << " <socket_path> <mode> [args...]\n\n";
        std::cerr << "Modes:\n";
        std::cerr << "  path <file>  - Predict from simulation_data.bin file\n";
        std::cerr << "  binary       - Predict from generated test data\n";
        std::cerr << "  shutdown     - Stop the server\n";
        std::cerr << "\nExamples:\n";
        std::cerr << "  " << argv[0] << " /tmp/boom.sock path data/simulations/run_xxx/simulation_data.bin\n";
        std::cerr << "  " << argv[0] << " /tmp/boom.sock binary\n";
        std::cerr << "  " << argv[0] << " /tmp/boom.sock shutdown\n";
        return 1;
    }

    const std::string socket_path = argv[1];
    const std::string mode = argv[2];

    try {
        BoomClient client(socket_path);

        if (mode == "path") {
            if (argc < 4) {
                std::cerr << "Error: path mode requires simulation file path\n";
                return 1;
            }
            const std::string sim_path = argv[3];
            std::cout << "Requesting prediction for: " << sim_path << "\n";

            BoomResult result = client.predict_path(sim_path);
            print_result(result);

            return result.ok ? 0 : 1;

        } else if (mode == "binary") {
            // Generate test data: 500 frames, 2000 pendulums
            const int frames = 500;
            const int pendulums = 2000;

            std::cout << "Generating test data: " << frames << " frames, "
                      << pendulums << " pendulums\n";

            std::vector<float> data(static_cast<size_t>(frames) * pendulums * 8);
            std::mt19937 rng(12345);
            std::uniform_real_distribution<float> dist(-3.14159f, 3.14159f);
            for (auto& v : data) {
                v = dist(rng);
            }

            size_t data_mb = data.size() * sizeof(float) / (1024 * 1024);
            std::cout << "Sending " << data_mb << " MB to server...\n";

            BoomResult result = client.predict_binary(data.data(), frames, pendulums);
            print_result(result);

            return result.ok ? 0 : 1;

        } else if (mode == "shutdown") {
            std::cout << "Requesting server shutdown...\n";
            client.shutdown();
            std::cout << "Shutdown request sent.\n";
            return 0;

        } else {
            std::cerr << "Unknown mode: " << mode << "\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
