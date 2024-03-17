from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

press_left_btn = 0
press_right_btn = 0

prev_xpos = 0
prev_ypos = 0

orbit_x = np.pi/4
orbit_y = np.pi/4
pan_x = 0.
pan_y = 0.
zoom_deg = 1.5
pers_zoom_deg = 1.5

proj_mode = 0

flip = 1
orbit_flip = 1

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program


def key_callback(window, key, scancode, action, mods):
    global proj_mode, zoom_deg, pers_zoom_deg
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        # change the projection mode
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                if proj_mode == 0:
                    proj_mode = 1
                    pers_zoom_deg = zoom_deg    # save current zoom degree
                    zoom_deg = 0                # no zoom in orthogonal mode
                else:
                    proj_mode = 0
                    zoom_deg = pers_zoom_deg    # restore zoom degree
                
                

def cursor_callback(window, xpos, ypos):
    global press_left_btn, press_right_btn, prev_xpos, prev_ypos, orbit_x, orbit_y, pan_x, pan_y, flip, orbit_flip
    # initial mouse position setting
    if prev_xpos == 0 and prev_ypos == 0:
        prev_xpos = xpos
        prev_ypos = ypos
    
    # orbit
    if press_left_btn == 1:
        orbit_x += (prev_xpos - xpos) * .003 * orbit_flip
        orbit_y += (ypos - prev_ypos) * .003
    # pan
    elif press_right_btn == 1:
        pan_x += (prev_xpos - xpos) * .002
        pan_y += (ypos - prev_ypos) * .002

    # when the view point is flipped by orbitting up/down
    if orbit_y % (2*np.pi) > np.pi/2 and orbit_y % (2*np.pi) < np.pi*3/2:
        flip = -1
    else:
        flip = 1

    prev_xpos = xpos
    prev_ypos = ypos

def button_callback(window, button, action, mod):
    global press_left_btn, press_right_btn, flip, orbit_flip
    # orbit
    if button==GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            press_left_btn = 1
            # change the orbit direction when the up-vector is flipped
            if flip == -1:
                orbit_flip = -1
            else:
                orbit_flip = 1
        elif action==GLFW_RELEASE:
            press_left_btn = 0
    # pan
    elif button==GLFW_MOUSE_BUTTON_RIGHT:
        if action==GLFW_PRESS:
            press_right_btn = 1
        elif action==GLFW_RELEASE:
            press_right_btn = 0

def scroll_callback(window, xoffset, yoffset):
    global zoom_deg, proj_mode
    # zoom
    if zoom_deg - yoffset * .1 > .1 or yoffset < 0:
        zoom_deg -= yoffset * .1
    else:
        zoom_deg = .1


def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -100.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         100.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, -100.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 100.0,  0.0, 0.0, 1.0, # z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         -200.0, 0.0, 0.0,  0.4, 0.4, 0.4,  # x-axis start
         200.0, 0.0, 0.0,  0.4, 0.4, 0.4,   # x-axis end 
         0.0, 0.0, -200.0,  0.4, 0.4, 0.4,  # z-axis start
         0.0, 0.0, 200.0,  0.4, 0.4, 0.4    # z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

# draw x-axis, z-axis
def draw_frame(vao, MVP, MVP_loc):
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, 6)

# draw grid
def draw_grid(vao, MVP, MVP_loc):
    for i in range(-200, 201):
        if i == 0:
            continue
        grid_MVP = MVP*glm.translate(glm.vec3(.1*i, 0, .1*i))
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(grid_MVP))
        glBindVertexArray(vao)
        glDrawArrays(GL_LINES, 0, 6)

def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls

    # create a window and OpenGL context
    window = glfwCreateWindow(950, 950, 'Basic OpenGL viewer - 2019092306 Koo Bonjun', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback)
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    MVP_loc = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_grid = prepare_vao_grid()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix 
        # orthogonal projection mode / perspective projection mode
        if proj_mode == 1:
            P = glm.ortho(-1,1,-1,1,-1,1)
        else:
            P = glm.perspective(45, 1, 0.1, 20)

        # viewing matrix
        # V1: orbit right/left, V2: orbit up/down, V3: pan
        V1 = glm.lookAt(glm.vec3(.1*np.sin(orbit_x),0,.1*np.cos(orbit_x)), glm.vec3(0,0,0), glm.vec3(0,1,0))
        V2 = glm.lookAt(glm.vec3(0,.1*np.sin(orbit_y),.1*np.cos(orbit_y)), glm.vec3(0,0,0), glm.vec3(0,flip,0))
        V3 = glm.lookAt(glm.vec3(pan_x,pan_y,zoom_deg), glm.vec3(pan_x,pan_y,-1), glm.vec3(0,1,0))
        V = V3*V2*V1
   
        # modeling matrix (default)
        M = glm.mat4()

        # current frame: P*V*M (now this is the world frame)
        MVP = P*V*M

        # draw current frame
        draw_frame(vao_frame, MVP, MVP_loc)

        # draw current grid
        draw_grid(vao_grid, MVP, MVP_loc)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
