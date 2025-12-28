/**
 * Boom detection C++ client example
 *
 * Demonstrates how to connect to the boom_server.py and send predictions.
 * Includes downsampling logic for large pendulum counts.
 *
 * Compile:
 *   g++ -std=c++17 -O2 boom_client_example.cpp -o boom_client
 *
 * Usage:
 *   # First start the Python server:
 *   uv run python scripts/boom_server.py models/boom_v1
 *
 *   # Then run client:
 *   ./boom_client /tmp/boom_server.sock path /path/to/simulation_data.bin
 *   ./boom_client /tmp/boom_server.sock binary  # (uses test data)
 */

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <stdexcept>

// -----------------------------------------------------------------------------
// Protocol helpers
// -----------------------------------------------------------------------------

class BoomClient {
public:
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
            throw std::runtime_error("Failed to connect to server at " + socket_path);
        }
    }

    ~BoomClient() {
        if (sock_ >= 0) {
            close(sock_);
        }
    }

    // Predict using a file path (server loads the file)
    std::string predict_path(const std::string& path) {
        std::string header = R"({"type":"path","path":")" + path + R"("})";
        return send_request(header, nullptr, 0);
    }

    // Predict using binary data (already downsampled)
    std::string predict_binary(const float* data, int frames, int pendulums, int values = 8) {
        std::string header = R"({"type":"binary","frames":)" + std::to_string(frames) +
                            R"(,"pendulums":)" + std::to_string(pendulums) +
                            R"(,"values":)" + std::to_string(values) + "}";

        size_t data_size = frames * pendulums * values * sizeof(float);
        return send_request(header, data, data_size);
    }

    // Shutdown the server
    void shutdown() {
        std::string header = R"({"type":"shutdown"})";
        send_request(header, nullptr, 0);
    }

private:
    int sock_ = -1;

    std::string send_request(const std::string& header, const void* data, size_t data_size) {
        // Send header length (4 bytes, little-endian)
        uint32_t header_len = static_cast<uint32_t>(header.size());
        send_all(&header_len, sizeof(header_len));

        // Send header
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
                throw std::runtime_error("Send failed");
            }
            sent += n;
        }
    }

    void recv_all(void* data, size_t size) {
        char* ptr = static_cast<char*>(data);
        size_t received = 0;
        while (received < size) {
            ssize_t n = recv(sock_, ptr + received, size - received, 0);
            if (n <= 0) {
                throw std::runtime_error("Recv failed");
            }
            received += n;
        }
    }

    std::string recv_response() {
        // Read response length
        uint32_t resp_len;
        recv_all(&resp_len, sizeof(resp_len));

        // Read response data
        std::string response(resp_len, '\0');
        recv_all(response.data(), resp_len);

        return response;
    }
};

// -----------------------------------------------------------------------------
// Downsampling for large pendulum counts
// -----------------------------------------------------------------------------

/**
 * Downsample simulation data from many pendulums to a fixed count.
 *
 * IMPORTANT: Use the same seed (42) as Python for reproducibility!
 *
 * @param src Source data, shape: [frames][src_pendulums][8]
 * @param frames Number of frames
 * @param src_pendulums Number of pendulums in source
 * @param dst Output buffer, shape: [frames][dst_pendulums][8]
 * @param dst_pendulums Target pendulum count (default: 2000)
 * @param seed Random seed (default: 42, matches Python PRODUCTION_CONFIG)
 */
void downsample_pendulums(
    const float* src,
    int frames,
    int src_pendulums,
    float* dst,
    int dst_pendulums = 2000,
    unsigned int seed = 42
) {
    if (src_pendulums <= dst_pendulums) {
        // No downsampling needed, just copy
        std::memcpy(dst, src, frames * src_pendulums * 8 * sizeof(float));
        return;
    }

    // Generate random indices (matching Python's numpy.random.RandomState)
    // Note: This uses a simple approach. For exact match with numpy,
    // you'd need to replicate numpy's MT19937 implementation.
    std::mt19937 rng(seed);

    // Fisher-Yates partial shuffle to select dst_pendulums indices
    std::vector<int> indices(src_pendulums);
    for (int i = 0; i < src_pendulums; ++i) {
        indices[i] = i;
    }

    for (int i = 0; i < dst_pendulums; ++i) {
        std::uniform_int_distribution<int> dist(i, src_pendulums - 1);
        int j = dist(rng);
        std::swap(indices[i], indices[j]);
    }

    // Sort selected indices for cache-friendly access
    std::sort(indices.begin(), indices.begin() + dst_pendulums);

    // Copy selected pendulums
    for (int f = 0; f < frames; ++f) {
        for (int p = 0; p < dst_pendulums; ++p) {
            int src_p = indices[p];
            const float* src_ptr = src + (f * src_pendulums + src_p) * 8;
            float* dst_ptr = dst + (f * dst_pendulums + p) * 8;
            std::memcpy(dst_ptr, src_ptr, 8 * sizeof(float));
        }
    }
}

// -----------------------------------------------------------------------------
// Example usage
// -----------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <socket_path> <mode> [path]\n";
        std::cerr << "  mode: 'path' (send file path) or 'binary' (send data)\n";
        std::cerr << "\nExamples:\n";
        std::cerr << "  " << argv[0] << " /tmp/boom_server.sock path data/simulations/run_xxx/simulation_data.bin\n";
        std::cerr << "  " << argv[0] << " /tmp/boom_server.sock binary\n";
        return 1;
    }

    std::string socket_path = argv[1];
    std::string mode = argv[2];

    try {
        BoomClient client(socket_path);

        if (mode == "path") {
            if (argc < 4) {
                std::cerr << "Error: path mode requires simulation file path\n";
                return 1;
            }
            std::string sim_path = argv[3];
            std::cout << "Requesting prediction for: " << sim_path << "\n";

            std::string response = client.predict_path(sim_path);
            std::cout << "Response: " << response << "\n";

        } else if (mode == "binary") {
            // Create test data: 100 frames, 5000 pendulums (will be downsampled to 2000)
            const int frames = 100;
            const int src_pendulums = 5000;
            const int dst_pendulums = 2000;

            std::cout << "Creating test data: " << frames << " frames, "
                      << src_pendulums << " pendulums\n";
            std::cout << "Downsampling to " << dst_pendulums << " pendulums\n";

            // Generate random test data
            std::vector<float> src_data(frames * src_pendulums * 8);
            std::mt19937 rng(123);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (auto& v : src_data) {
                v = dist(rng);
            }

            // Downsample
            std::vector<float> dst_data(frames * dst_pendulums * 8);
            downsample_pendulums(
                src_data.data(), frames, src_pendulums,
                dst_data.data(), dst_pendulums,
                42  // Same seed as Python!
            );

            std::cout << "Sending " << (dst_data.size() * sizeof(float) / 1024 / 1024)
                      << " MB to server...\n";

            std::string response = client.predict_binary(
                dst_data.data(), frames, dst_pendulums
            );
            std::cout << "Response: " << response << "\n";

        } else if (mode == "shutdown") {
            std::cout << "Shutting down server...\n";
            client.shutdown();

        } else {
            std::cerr << "Unknown mode: " << mode << "\n";
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
