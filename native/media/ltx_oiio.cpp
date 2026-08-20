#include <OpenImageIO/imageio.h>

#include <cstddef>
#include <exception>
#include <memory>
#include <string>

namespace {
thread_local std::string last_error;

int fail(const std::string& message) {
    last_error = message;
    return 1;
}
}

#if defined(_WIN32)
#define LTX_MEDIA_EXPORT __declspec(dllexport)
#else
#define LTX_MEDIA_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

LTX_MEDIA_EXPORT const char* ltx_oiio_version() {
    return OIIO_VERSION_STRING;
}

LTX_MEDIA_EXPORT const char* ltx_oiio_last_error() {
    return last_error.c_str();
}

LTX_MEDIA_EXPORT int ltx_oiio_image_info(const char* path, int* width, int* height, int* channels) {
    try {
        auto input = OIIO::ImageInput::open(path);
        if (!input) {
            return fail(OIIO::geterror());
        }
        const auto& spec = input->spec();
        *width = spec.width;
        *height = spec.height;
        *channels = spec.nchannels;
        input->close();
        last_error.clear();
        return 0;
    } catch (const std::exception& exception) {
        return fail(exception.what());
    }
}

LTX_MEDIA_EXPORT int ltx_oiio_read_float(const char* path, float* pixels, int element_capacity) {
    try {
        auto input = OIIO::ImageInput::open(path);
        if (!input) {
            return fail(OIIO::geterror());
        }
        const auto& spec = input->spec();
        const auto required = static_cast<std::size_t>(spec.width) * spec.height * spec.nchannels;
        if (required > static_cast<std::size_t>(element_capacity)) {
            return fail("destination buffer is too small");
        }
        if (!input->read_image(0, 0, 0, -1, OIIO::TypeDesc::FLOAT, pixels)) {
            return fail(input->geterror());
        }
        input->close();
        last_error.clear();
        return 0;
    } catch (const std::exception& exception) {
        return fail(exception.what());
    }
}

LTX_MEDIA_EXPORT int ltx_oiio_write_float(
    const char* path, int width, int height, int channels, const float* pixels) {
    try {
        auto output = OIIO::ImageOutput::create(path);
        if (!output) {
            return fail(OIIO::geterror());
        }
        OIIO::ImageSpec spec(width, height, channels, OIIO::TypeDesc::HALF);
        spec.attribute("oiio:ColorSpace", "scene_linear");
        if (!output->open(path, spec)) {
            return fail(output->geterror());
        }
        if (!output->write_image(OIIO::TypeDesc::FLOAT, pixels)) {
            return fail(output->geterror());
        }
        output->close();
        last_error.clear();
        return 0;
    } catch (const std::exception& exception) {
        return fail(exception.what());
    }
}

}
