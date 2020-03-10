#include <stdio.h>
#include <math.h>
#include "aruco.h"
#include "cvdrawingutils.h"
#include <fstream>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <sstream>
#include <string>
#include <stdexcept>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>

#include "shader.h"
#include "opengl_tools.h"

enum Stereoscopy {
	No,
	Toe_In,
	Off_Axis
};

enum Eye {
	Center,
	Left,
	Right
};

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void render(ToolsC *tools, Shader shader, Eye eye);
glm::mat4 calculate_off_axis_frustrum(Eye eye);

// settings
const unsigned int SCR_WIDTH = 1280;
const unsigned int SCR_HEIGHT = 720;

#ifdef _WIN32
std::string BASE_PATH = "../../";
#else
std::string BASE_PATH = "../../";
#endif

// aruco
float f = 0.0;

aruco::MarkerDetector MDetector;
cv::VideoCapture TheVideoCapturer;
std::vector<aruco::Marker> TheMarkers;
cv::Mat TheInputImage, TheInputImageGrey, TheInputImageCopy;
aruco::CameraParameters TheCameraParameters;
int iDetectMode = 0, iMinMarkerSize = 0, iCorrectionRate = 0, iShowAllCandidates = 0, iEnclosed = 0, iThreshold, iCornerMode, iDictionaryIndex, iTrack = 0;

class CmdLineParser { int argc; char** argv; public:CmdLineParser(int _argc, char** _argv) : argc(_argc), argv(_argv) {}   bool operator[](std::string param) { int idx = -1;  for (int i = 0; i < argc && idx == -1; i++)if (std::string(argv[i]) == param)idx = i; return (idx != -1); }    std::string operator()(std::string param, std::string defvalue = "-1") { int idx = -1; for (int i = 0; i < argc && idx == -1; i++)if (std::string(argv[i]) == param)idx = i; if (idx == -1)return defvalue; else return (argv[idx + 1]); } };
struct   TimerAvrg { std::vector<double> times; size_t curr = 0, n; std::chrono::high_resolution_clock::time_point begin, end;   TimerAvrg(int _n = 30) { n = _n; times.reserve(n); }inline void start() { begin = std::chrono::high_resolution_clock::now(); }inline void stop() { end = std::chrono::high_resolution_clock::now(); double duration = double(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())*1e-6; if (times.size() < n) times.push_back(duration); else { times[curr] = duration; curr++; if (curr >= times.size()) curr = 0; } }double getAvrg() { double sum = 0; for (auto t : times) sum += t; return sum / double(times.size()); } };
TimerAvrg Fps;

// camera viewpoint
glm::vec3 current_pos = glm::vec3(0.0f, 0.0f, -0.5f);
glm::mat4 m_view = glm::translate(glm::mat4(1.0f), current_pos);
glm::vec4 orientation = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
glm::mat4 m_view_segundo_cubo = glm::translate(glm::mat4(1.0f), glm::vec3(0.2f, 0.0f, 0.0f));

// Stereoscopy
Stereoscopy steroscopy = Stereoscopy::No;

int main(int argc, char **argv)
{
	ToolsC * tools = new ToolsC(BASE_PATH);

	CmdLineParser cml(argc, argv);
	// read camera parameters if passed
	if (cml["-c"])
		TheCameraParameters.readFromXMLFile(cml("-c"));

	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "RV – Practica tracking", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	// glad: load all OpenGL function pointers
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	// configure global opengl state
	glEnable(GL_DEPTH_TEST);

	// build and compile shaders
	Shader ourShader(std::string(BASE_PATH + "vert.vs").c_str(), std::string(BASE_PATH + "frag.fs").c_str());
	Shader ourShader2D(std::string(BASE_PATH + "vert2D.vs").c_str(), std::string(BASE_PATH + "frag2D.fs").c_str());

	// load and create a texture 
	tools->loadTextures();

	// initializes vertex buffers
	tools->initRenderData();

	std::cout << "VBOs[0] = " << tools->m_VBOs[0] << "\n";
	std::cout << "VAOs[0] = " << tools->m_VAOs[0] << "\n";
	std::cout << "VBOs[1] = " << tools->m_VBOs[1] << "\n";
	std::cout << "VAOs[1] = " << tools->m_VAOs[1] << "\n";
	std::cout << "VBOs[2] = " << tools->m_VBOs[2] << "\n";
	std::cout << "VAOs[2] = " << tools->m_VAOs[2] << "\n";

	// set up shader materials
	ourShader.use();
	ourShader.setInt("material.diffuse", 0);
	ourShader.setInt("material.specular", 1);

	// set up shader materials
	ourShader2D.use();
	ourShader2D.setInt("image", 1);

	// opens video input from webcam
	TheVideoCapturer.open(0);

	// check video is open
	if (!TheVideoCapturer.isOpened())
		throw std::runtime_error("Could not open video");

	// render loop
	while (!glfwWindowShouldClose(window))
	{
		// this will contain the image from the webcam
		cv::Mat frame, frameCopy;

		// capture the next frame from the webcam
		TheVideoCapturer >> frame;
		cv::cvtColor(frame, frame, CV_RGB2BGR);
		std::cout << "Frame size: " << frame.cols << " x " << frame.rows << " x " << frame.channels() << "\n";

		// creates a copy of the grabbed frame
		frame.copyTo(frameCopy);

		// Tamaño en cm del marcador
		float TheMarkerSize = 0.1;

		if (TheCameraParameters.isValid())
			std::cout << "Parameters OK\n";

		// TAREA 1: detectar marcadores con MDetector.detect (ver librería Aruco)
		TheMarkers = MDetector.detect(frameCopy, TheCameraParameters, TheMarkerSize, true);

		// TAREA 1: para cada marcador, dibujar la imagen frameCopy
		//	- el marcador en la imagen usando método draw() de aruco::Marker
		//	- el eje local de coordenadas del marcador con aruco::CvDrawingUtils::draw3dAxis()
		for (unsigned int i = 0; i < TheMarkers.size(); i++)
		{
			aruco::CvDrawingUtils::draw3dAxis(frameCopy, TheMarkers[i], TheCameraParameters, 5);
			TheMarkers[i].draw(frameCopy);
		}

		// copies input image to m_textures[1]
		flip(frameCopy, frameCopy, 0);
		glBindTexture(GL_TEXTURE_2D, tools->m_textures[1]);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frameCopy.cols, frameCopy.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, frameCopy.ptr());

		// input
		processInput(window);

		// render
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // also clear the depth buffer now!
		if (steroscopy == Stereoscopy::No) {
			render(tools, ourShader, Eye::Center);
		}
		else {
			glColorMask(true, false, false, false);

			render(tools, ourShader, Eye::Left);

			glClear(GL_DEPTH_BUFFER_BIT);

			glColorMask(false, true, true, false);
			render(tools, ourShader, Eye::Right);

			glColorMask(true, true, true, true);
		}

		ourShader2D.use();
		float screen_width = 2.0f;
		glm::mat4 projection2D = glm::ortho(0.0f, (float)SCR_WIDTH, 0.0f, (float)SCR_HEIGHT, -1.0f, 1.0f);
		glm::mat4 model2D = glm::translate(glm::mat4(1.0f), glm::vec3((float)SCR_WIDTH / 2.0f - 160.0f, 540.0f, 0.f));

		ourShader2D.setMat4("projection2D", projection2D);
		//std::cout << "glGetUniformLocation " << " :" << glGetUniformLocation(ourShader2D.ID, "projection2D");
		ourShader2D.setMat4("model2D", model2D);

		glBindVertexArray(tools->m_VAOs[0]);                 //VAOs[0] is 2D quad for cam input
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glBindVertexArray(0);

		glUseProgram(0);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// de-allocate all resources once they've outlived their purpose:
	glDeleteVertexArrays(1, &(tools->m_VAOs[2]));
	glDeleteBuffers(1, &(tools->m_VBOs[2]));

	glDeleteVertexArrays(1, &(tools->m_VAOs[1]));
	glDeleteBuffers(1, &(tools->m_VBOs[1]));

	glDeleteVertexArrays(1, &(tools->m_VAOs[0]));
	glDeleteBuffers(1, &(tools->m_VBOs[0]));

	// glfw: terminate, clearing all previously allocated GLFW resources.
	glfwTerminate();

	std::cout << "Bye!" << std::endl;
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
		steroscopy = Stereoscopy::No;
	}
	if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
		steroscopy = Stereoscopy::Toe_In;
	}
	if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
		steroscopy = Stereoscopy::Off_Axis;
	}
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

void render(ToolsC *tools, Shader shader, Eye eye) {
	// bind textures on corresponding texture units
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tools->m_textures[0]);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, tools->m_textures[1]);

	// activate shader
	shader.use();

	// create transformations
	glm::mat4 current_view = glm::mat4(1.0f);
	glm::mat4 projection;

	if (steroscopy == Stereoscopy::Off_Axis) {
		projection = calculate_off_axis_frustrum(eye);
	}
	else {
		projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
	}
	// pass transformation matrices to the shader
	shader.setMat4("projection", projection);

	if (TheMarkers.size() > 0)
	{
		// TAREA 2
		float pos_x = TheMarkers[0].Tvec.at<float>(0, 0);
		float pos_y = TheMarkers[0].Tvec.at<float>(1, 0);
		float pos_z = TheMarkers[0].Tvec.at<float>(2, 0);
		current_pos = glm::vec3(pos_x, pos_y, -pos_z);

		// TAREA 3
		glm::mat4 rotation_matrix = glm::mat4(1.0f);
		float rot_z = TheMarkers[0].Rvec.at<float>(1, 0);

		rotation_matrix = glm::rotate(rotation_matrix, rot_z, glm::vec3(0.0f, 0.0f, 1.0f));
		glm::vec4 normal = glm::vec4(0.0f, -1.0f, 0.0f, 1.0f);
		orientation = glm::vec4(rotation_matrix * normal);

		// TAREA 4
		if (TheMarkers.size() > 1) {
			// TODO: Poner el indice 1
			float pos_x_other_marker = TheMarkers[1].Tvec.at<float>(0, 0);
			float pos_y_other_marker = TheMarkers[1].Tvec.at<float>(1, 0);
			float pos_z_other_marker = TheMarkers[1].Tvec.at<float>(2, 0);

			m_view_segundo_cubo[3][0] = pos_x_other_marker;
			m_view_segundo_cubo[3][1] = pos_y_other_marker;
			m_view_segundo_cubo[3][2] = -pos_z_other_marker;
		}
	}
	else
	{
		std::cout << "No marker detected\n";
	}

	glm::vec3 camera_target = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::vec3 camera_pos = current_pos;

	float eye_separation = 0.065f;
	if (steroscopy != Stereoscopy::No) {
		int sign = (eye == Eye::Right) ? 1 : -1;
		camera_pos.x = current_pos.x + sign * eye_separation / 2;
	}

	m_view = glm::lookAt(camera_pos, camera_target, glm::vec3(orientation.x, orientation.y, orientation.z));

	shader.setMat4("view", m_view);
	shader.setVec3("viewPos", camera_pos);
	shader.setVec3("light.direction", -1.0f, .0f, -1.0f);

	// light properties
	shader.setVec3("light.ambient", 0.5f, 0.5f, 0.5f);
	shader.setVec3("light.diffuse", 0.75f, 0.75f, 0.75f);
	shader.setVec3("light.specular", 1.0f, 1.0f, 1.0f);

	// material properties
	shader.setFloat("material.shininess", 1.0f);

	// render object
	glBindVertexArray(tools->m_VAOs[2]);

	// calculate the model matrix for each object and pass it to shader before drawing
	glm::mat4 model = glm::mat4(1.0f);

	// sets model matrix
	shader.setMat4("model", model);

	// draws
	glDrawArrays(GL_TRIANGLES, 0, 36);

	// Pintar segundo cubo
	shader.setMat4("model", m_view_segundo_cubo);
	glDrawArrays(GL_TRIANGLES, 0, 36);

	glBindVertexArray(0);

	shader.setMat4("model", model);
	shader.setInt("material.specular", 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, tools->m_textures[2]);
	glBindVertexArray(tools->m_VAOs[1]);                 //VAOs[1] is floor
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);

	glUseProgram(0);

	// Prepare transformations
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, tools->m_textures[1]);
}

glm::mat4 calculate_off_axis_frustrum(Eye eye) {
	float eye_separation = 0.065f;
	float convergence = 1.0f;
	float aspect_ratio = 1.3333f;
	float fov = 45.0f;
	float near_clip_plane = 0.01f;
	float far_clip_plane = 10.0f;

	float top, bottom, left, right;
	float a, b, c;

	top = near_clip_plane * tan(fov / 2);
	bottom = -top;

	a = aspect_ratio * tan(fov / 2) * convergence;
	b = a - eye_separation / 2;
	c = a + eye_separation / 2;
	if (eye == Eye::Left) {
		left = -b * near_clip_plane / convergence;
		right = c * near_clip_plane / convergence;
	}
	else {
		left = -c * near_clip_plane / convergence;
		right = b * near_clip_plane / convergence;
	}


	return glm::frustum(left, right, bottom, top, near_clip_plane, far_clip_plane);
}