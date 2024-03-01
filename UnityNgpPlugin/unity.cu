
#include <iostream>
#include <GL/gl3w.h>
#include <GLFW/glfw3.h>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/dlss.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <memory>
#include <Eigen>

#define API __stdcall
#define EXPORT __declspec(dllexport)

// !!! TODO: REMOVE #define NGP_UI from render_buffer.h  !!!



using RenderingEvent = void(__stdcall*)(int event_id);
using Texture = std::shared_ptr<ngp::GLTexture>;
using RenderBuffer = std::shared_ptr<ngp::CudaRenderBuffer>;


struct TextureHandles {
	GLuint color_buffer;
	GLuint depth_buffer;
};

struct TextureData {
	TextureData(
		const Texture& c_tex,
		const Texture& d_tex,
		const RenderBuffer& c_rb,
		int widht, int height
	) :
		color_texture(c_tex),
		depth_texture(d_tex),
		render_buffer(c_rb),
		width(widht),
		height(height)
	{}

	Texture color_texture;
	Texture depth_texture;
	RenderBuffer render_buffer;
	int width;
	int height;
};

std::shared_ptr<TextureData> textures;

const int INIT_EVENT = 0x0001;
const int DRAW_EVENT = 0x0002;
const int DESTROY_EVENT = 0x0003;
const int CREATE_TEXTURE_EVENT = 0x0004;
const int DESTROY_VULKAN_EVENT = 0x0005;


bool graphics_initialized = false;
bool use_dlss = false;
static int _width;
static int _height;

float zero_matrix[12] = { 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f };
float* view_matrix;



static TextureHandles texture_handles;


static std::shared_ptr<ngp::Testbed> testbed;
std::shared_ptr<ngp::IDlssProvider> dlss_provider;


static void API on_render_thread(int event_id)
{
	switch (event_id)
	{
	case INIT_EVENT:
		init();
		break;
	case DRAW_EVENT:
		update_texture();
		break;
	case DESTROY_EVENT:
		destroy_texture();
		destroy();
		break;
	case CREATE_TEXTURE_EVENT:
		texture_handles = create_texture(_width, _height);

		break;
	case DESTROY_VULKAN_EVENT:
		destroy_vulkan();
		break;
	}
}

static void API init()
{

	//Init NGP

}


extern "C" void EXPORT API destroy_texture()
{
	if(!testbed)
	{
		std::cout << "Testbed not initialized" << std::endl;
		return;
	}
	auto texture = textures;

	if (texture == nullptr) {
		std::cout << "Texture not found" << std::endl;
		return;
	}

	texture->render_buffer->reset_accumulation();
	texture->render_buffer.reset();
	texture.reset();

	std::cout << "GLTexture and render buffer destroyed" << std::endl;


	
}

static void API destroy()
{

	testbed.reset();
	texture_handles.color_buffer = 0;
	texture_handles.depth_buffer = 0;
	glfwTerminate();

	view_matrix = nullptr;
	graphics_initialized = false;
	use_dlss = false;
}

static TextureHandles API create_texture(int width, int height)
{
	if (!testbed) {
		std::cout << "Testbed not initialized" << std::endl;
		return {};
	}

	auto c_texture = std::make_shared<ngp::GLTexture>();
	auto d_texture = std::make_shared<ngp::GLTexture>();
	auto render_buffer = std::make_shared<ngp::CudaRenderBuffer>(c_texture, d_texture);
	Eigen::Vector2i render_resolution{ width, height };

#if defined(NGP_VULKAN)
	if (use_dlss) {
		render_buffer->enable_dlss(*dlss_provider, render_resolution);


		Eigen::Vector2i texture_resolution{ width, height };
		render_resolution = render_buffer->in_resolution();
		if (render_resolution.isZero()) {
			render_resolution = texture_resolution / 16;
		}
		else {
			render_resolution = render_resolution.cwiseMin(texture_resolution);
		}

		if (render_buffer->dlss()) {
			render_resolution = render_buffer->dlss()->clamp_resolution(render_resolution);
		}



	}
	else {
		render_buffer->disable_dlss();
	}
#endif // 

	render_buffer->resize(render_resolution);

	GLuint c_handle = c_texture->texture();
	GLuint d_handle = d_texture->texture();

	textures = std::make_shared<TextureData>(
		c_handle,
		d_handle,
		render_buffer,
		width,
		height
	);

	return { c_handle, d_handle };

}

void API update_texture() {
	if (!testbed)
	{
		return;
	}

	auto color = textures;


	if (color == nullptr) {
		std::cout << " texture not found" << std::endl;
		return;
	}

	Eigen::Matrix<float, 3, 4> camera{ view_matrix };

	RenderBuffer render_buffer = color->render_buffer;
	

	render_buffer->reset_accumulation();
	
	testbed->render_frame(
		testbed->m_stream.get(),
		camera,
		camera,
		camera,
		Eigen::Vector4f::Zero(),
		testbed->m_relative_focal_length,
		Eigen::Vector4f::Zero(),
		{},
		{},
		testbed->m_visualized_dimension,
		*render_buffer,
		true,
		&(testbed->primary_device())
	);


}

static void API destroy_vulkan()
{
#ifdef NGP_VULKAN    
	if (use_dlss) {
		dlss_provider.reset();
		use_dlss = false;
	}
#endif
}

//external C load scene file
extern "C" EXPORT void API setup_initialization_params(
	const char* scene,
	const char* snapshot,
	bool dlss,
	int width,
	int height
)
{
	use_dlss = dlss;
	_width = width;
	_height = height;

	testbed = std::make_shared<ngp::Testbed>(ngp::ETestbedMode::Nerf, scene);

	if (snapshot)
		testbed->load_snapshot(snapshot);
}

static void API init_graphics()
{
	if (graphics_initialized) return;

	if (!glfwInit())
	{
		std::cout << "Failed to initialize GLFW" << std::endl;
	}
	if (!gl3wInit())
	{
		std::cout << "Failed to initialize GL3W" << std::endl;
	}

	//#ifdef NGP_VULKAN

	if (use_dlss) {
		try {
			dlss_provider = ngp::init_vulkan_and_ngx();
		}
		catch (const std::exception& e) {
			std::cout << "Failed to initilize Vulkan" << e.what() << std::endl;
			use_dlss = false;
		}
	}

	//#else
	use_dlss = false;
	//#endif

	graphics_initialized = true;
}


extern "C" RenderingEvent EXPORT API GetRenderEventFunc()
{
	return on_render_thread;
}


extern "C" EXPORT GLuint API get_color_buffer_handle() {

	return texture_handles.color_buffer;
}

extern "C" EXPORT GLuint API get_depth_buffer_handle() {

	return texture_handles.depth_buffer;
}


extern "C" EXPORT void API udpate_veiw_matrix(float* viewmatrix) {

	view_matrix = viewmatrix;
}

extern "C" EXPORT bool API is_graphics_initialized() {
	return graphics_initialized;
}
