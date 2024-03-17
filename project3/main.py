from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

# variables for view point
press_left_btn = 0
press_right_btn = 0
prev_xpos = 0
prev_ypos = 0
orbit_x = np.pi/4
orbit_y = np.pi/4
pan_x = 0.
pan_y = 0.
zoom_deg = 8
pers_zoom_deg = 8
proj_mode = 0
flip = 1
orbit_flip = 1

# variables for dropped bvh file
vao_drop = 0
vbo_drop = 0
hierarchy_drop = 0
is_paused = 1
rendering_mode = 1

g_vertex_shader_src_box= '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(transpose(inverse(M))) * vin_normal);
}
'''

g_fragment_shader_src_box = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;

out vec4 FragColor;

uniform vec3 view_pos;
uniform vec3 mat_color;

void main()
{
    // first light source

    // light and material properties
    vec3 light_pos1 = vec3(10,10,10);
    vec3 light_pos2 = vec3(-10,10,-10);
    vec3 light_color = vec3(.8,.8,.8);
    vec3 material_color = mat_color;
    float material_shininess = 128;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = material_color;  // or can be light_color

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir1 = normalize(light_pos1 - surface_pos);
    vec3 light_dir2 = normalize(light_pos2 - surface_pos);

    // diffuse
    float diff1 = max(dot(normal, light_dir1), 0);
    float diff2 = max(dot(normal, light_dir2), 0);
    vec3 diffuse1 = diff1 * light_diffuse * material_diffuse;
    vec3 diffuse2 = diff2 * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir1 = reflect(-light_dir1, normal);
    vec3 reflect_dir2 = reflect(-light_dir2, normal);
    float spec1 = pow( max(dot(view_dir, reflect_dir1), 0.0), material_shininess);
    float spec2 = pow( max(dot(view_dir, reflect_dir2), 0.0), material_shininess);
    vec3 specular1 = spec1 * light_specular * material_specular;
    vec3 specular2 = spec2 * light_specular * material_specular;

    vec3 color = ambient + diffuse1 + diffuse2 + specular1 + specular2;

    FragColor = vec4(color, 1.);
}
'''

g_vertex_shader_src_line = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 

uniform mat4 MVP;

void main()
{
    gl_Position = MVP * vec4(vin_pos, 1.0);
}
'''

g_fragment_shader_src_line = '''
#version 330 core

out vec4 FragColor;

uniform vec3 color;

void main()
{
    FragColor = vec4(color, 1.0);
}
'''

g_vertex_shader_src_grid = '''
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

g_fragment_shader_src_grid = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
}
'''


class Node:
    # variables for bvh motion
    motion_index = 0
    motion = []
    frame_time = 0
    scailing = 1

    def __init__(self, parent, type, name, link_transform, joint_transform_list):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        self.name = name
        self.type = type

        # transform
        self.joint_transform_list = joint_transform_list
        self.joint_transform = glm.mat4()
        self.link_transform = link_transform
        self.global_transform = glm.mat4()

        # shape
        self.shape_transform = glm.mat4()

    def set_shape_transform(self):
        if self.parent is not None:
            link_vector = self.get_point() - self.parent.get_point()
            self.shape_transform = box_rotation(link_vector) * glm.translate((0,glm.length(link_vector)/2,0)) * glm.scale((.05,glm.length(link_vector)/3,.05))
        for child in self.children:
            child.set_shape_transform()
    
    def set_joint_transform(self):
        joint_transform = glm.mat4()
        Node.motion_index %= len(Node.motion)
        for i in range(len(self.joint_transform_list)):
            j = Node.motion_index + i
            cur_trans = self.joint_transform_list[i]
            if cur_trans == "XPOSITION" or cur_trans == "Xposition":
                joint_transform *= glm.translate((Node.motion[j]*Node.scailing,0,0))
            elif cur_trans == "YPOSITION" or cur_trans == "Yposition":
                joint_transform *= glm.translate((0,Node.motion[j]*Node.scailing,0))
            elif cur_trans == "ZPOSITION" or cur_trans == "Zposition":
                joint_transform *= glm.translate((0,0,Node.motion[j]*Node.scailing))
            elif cur_trans == "XROTATION" or cur_trans == "Xrotation":
                joint_transform *= glm.rotate(np.radians(Node.motion[j]), (1,0,0))
            elif cur_trans == "YROTATION" or cur_trans == "Yrotation":
                joint_transform *= glm.rotate(np.radians(Node.motion[j]), (0,1,0))
            elif cur_trans == "ZROTATION" or cur_trans == "Zrotation":
                joint_transform *= glm.rotate(np.radians(Node.motion[j]), (0,0,1))
        
        self.joint_transform = joint_transform
        Node.motion_index += len(self.joint_transform_list)

        for child in self.children:
            child.set_joint_transform()

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.link_transform * self.joint_transform
        else:
            self.global_transform = self.link_transform * self.joint_transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_point(self):
        return self.global_transform*glm.vec4(0,0,0,1).xyz

    def get_points(self):
        total_vertex = 0
        if self.parent is not None:
            points = [self.parent.global_transform*glm.vec4(0,0,0,1).xyz, self.global_transform*glm.vec4(0,0,0,1).xyz]
            total_vertex += 2
        else:
            points = []
        for child in self.children:
            child_points, child_total_vertex = child.get_points()
            points.extend(child_points)
            total_vertex += child_total_vertex
        return points, total_vertex
    
    def get_parent(self):
        return self.parent

    def get_children(self):
        return self.children

    def get_global_transform(self):
        return self.global_transform

    def get_shape_transform(self):
        return self.shape_transform


def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------

    # vertex shader
    # create an empty shader object
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    # provide shader source code
    glShaderSource(vertex_shader, vertex_shader_source)
    # compile the shader object
    glCompileShader(vertex_shader)

    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())

    # fragment shader
    # create an empty shader object
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    # provide shader source code
    glShaderSource(fragment_shader, fragment_shader_source)
    # compile the shader object
    glCompileShader(fragment_shader)

    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    # create an empty program object
    shader_program = glCreateProgram()
    # attach the shader objects to the program object
    glAttachShader(shader_program, vertex_shader)
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
    global proj_mode, rendering_mode, zoom_deg, pers_zoom_deg, is_paused
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        # change the projection mode
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            if key == GLFW_KEY_V:
                if proj_mode == 0:
                    pers_zoom_deg = zoom_deg    # save current zoom degree
                    zoom_deg = 0                # no zoom in orthogonal mode
                else:
                    zoom_deg = pers_zoom_deg    # restore zoom degree
                proj_mode = (proj_mode + 1) % 2
            elif key == GLFW_KEY_1:
                rendering_mode = 1
            elif key == GLFW_KEY_2:
                rendering_mode = 2
            elif key == GLFW_KEY_SPACE:
                is_paused = (is_paused + 1) % 2


def cursor_callback(window, xpos, ypos):
    global press_left_btn, press_right_btn, prev_xpos, prev_ypos, orbit_x, orbit_y, pan_x, pan_y, flip, orbit_flip, zoom_deg, proj_mode
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
        sensitivity = 0
        if proj_mode == 1:
            sensitivity = .002
        else:
            sensitivity = .001 * zoom_deg

        pan_x += (prev_xpos - xpos) * sensitivity
        pan_y += (ypos - prev_ypos) * sensitivity

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
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            press_left_btn = 1
            # change the orbit direction when the up-vector is flipped
            if flip == -1:
                orbit_flip = -1
            else:
                orbit_flip = 1
        elif action == GLFW_RELEASE:
            press_left_btn = 0
    # pan
    elif button == GLFW_MOUSE_BUTTON_RIGHT:
        if action == GLFW_PRESS:
            press_right_btn = 1
        elif action == GLFW_RELEASE:
            press_right_btn = 0


def scroll_callback(window, xoffset, yoffset):
    global zoom_deg, proj_mode
    # zoom
    if zoom_deg - yoffset > .1 or yoffset < 0:
        zoom_deg -= yoffset
    else:
        zoom_deg = .1


def drop_callback(window, paths):
    global hierarchy_drop, vao_drop, vbo_drop, is_paused
    is_paused = 1
    joint_num = 0
    joint_list = []
    path = paths[0]
    stack = []
    max_height = -100000
    min_height = 100000
    
    with open(path, 'r') as bvhfile:
        content = bvhfile.read()
    
    words = content.split()

    for i in range(len(words)):
        if words[i] == "ROOT" or words[i] == "JOINT" or words[i] == "End":
            cur_height = 0.0
            if len(stack) == 0:
                cur_height = float(words[i+5])
            else:
                cur_height = stack[-1] + float(words[i+5])
            max_height = max(max_height, cur_height)
            min_height = min(min_height, cur_height)
            stack.append(cur_height)
        elif words[i] == "}":
            stack.pop()
    scaling = 3 / (max_height - min_height)

    for i in range(len(words)):
        if words[i] == "ROOT":
            link_transform = glm.translate((float(words[i+4])*scaling,float(words[i+5])*scaling,float(words[i+6])*scaling))
            joint_transform_list = words[i+9:i+9+int(words[i+8])]
            hierarchy_drop = Node(None, words[i], words[i+1], link_transform, joint_transform_list)
            stack.append(hierarchy_drop)
            joint_num += 1
            joint_list.append(words[i+1])
        elif words[i] == "JOINT":
            link_transform = glm.translate((float(words[i+4])*scaling,float(words[i+5])*scaling,float(words[i+6])*scaling))
            joint_transform_list = words[i+9:i+9+int(words[i+8])]
            node = Node(stack[-1], words[i], words[i+1], link_transform, joint_transform_list)
            stack.append(node)
            joint_num += 1
            joint_list.append(words[i+1])
        elif words[i] == "End":
            link_transform = glm.translate((float(words[i+4])*scaling,float(words[i+5])*scaling,float(words[i+6])*scaling))
            end = Node(stack[-1], words[i], words[i+1], link_transform, [])
            stack.append(end)
        elif words[i] == "}":
            stack.pop()

    motion_loc = words.index("MOTION")

    hierarchy_drop.update_tree_global_transform()
    points, total_vertex = hierarchy_drop.get_points()
    hierarchy_drop.set_shape_transform()

    Node.motion_index = 0
    Node.scailing = scaling
    Node.frame_time = float(words[motion_loc+5])
    Node.motion = list(map(float, words[motion_loc+6:]))

    # print bvh file information
    print("<bvh file information>")
    print("File name: %s" % path)
    print("Number of frames: " + str(words[motion_loc+2]))
    print("FPS: %.2f" % (1/float(words[motion_loc+5])))
    print("Number of joints: " + str(joint_num))
    print("List of all joint names: " + str(joint_list))
    print()

    vao_drop, vbo_drop = initialize_vao_for_points(points)
    copy_points_data(points, vbo_drop)

def initialize_vao_for_points(points):
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # only allocate VBO and not copy data by specifying the third argument to None
    vertices = glm.array(points)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, None, GL_DYNAMIC_DRAW)

    # configure vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # return VBO along with VAO as it is needed when copying updated point position to VBO
    return VAO, VBO

def copy_points_data(points, vbo):
    glBindBuffer(GL_ARRAY_BUFFER, vbo)  # activate VBO

    # prepare vertex data (in main memory)
    vertices = glm.array(points)

    # only copy vertex data to VBO and not allocating it
    # glBufferSubData(taraget, offset, size, data)
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices.ptr)

def prepare_vao_box():
    # prepare vertex data (in main memory)
    # 36 vertices for 12 triangles
    vertices = glm.array(glm.float32,
        # position      normal
        -1 ,  1 ,  1 ,  0, 0, 1, # v0
         1 , -1 ,  1 ,  0, 0, 1, # v2
         1 ,  1 ,  1 ,  0, 0, 1, # v1

        -1 ,  1 ,  1 ,  0, 0, 1, # v0
        -1 , -1 ,  1 ,  0, 0, 1, # v3
         1 , -1 ,  1 ,  0, 0, 1, # v2

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 ,  1 , -1 ,  0, 0,-1, # v5
         1 , -1 , -1 ,  0, 0,-1, # v6

        -1 ,  1 , -1 ,  0, 0,-1, # v4
         1 , -1 , -1 ,  0, 0,-1, # v6
        -1 , -1 , -1 ,  0, 0,-1, # v7

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 ,  1 ,  0, 1, 0, # v1
         1 ,  1 , -1 ,  0, 1, 0, # v5

        -1 ,  1 ,  1 ,  0, 1, 0, # v0
         1 ,  1 , -1 ,  0, 1, 0, # v5
        -1 ,  1 , -1 ,  0, 1, 0, # v4
 
        -1 , -1 ,  1 ,  0,-1, 0, # v3
         1 , -1 , -1 ,  0,-1, 0, # v6
         1 , -1 ,  1 ,  0,-1, 0, # v2

        -1 , -1 ,  1 ,  0,-1, 0, # v3
        -1 , -1 , -1 ,  0,-1, 0, # v7
         1 , -1 , -1 ,  0,-1, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 ,  1 ,  1, 0, 0, # v2
         1 , -1 , -1 ,  1, 0, 0, # v6

         1 ,  1 ,  1 ,  1, 0, 0, # v1
         1 , -1 , -1 ,  1, 0, 0, # v6
         1 ,  1 , -1 ,  1, 0, 0, # v5

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 , -1 , -1 , -1, 0, 0, # v7
        -1 , -1 ,  1 , -1, 0, 0, # v3

        -1 ,  1 ,  1 , -1, 0, 0, # v0
        -1 ,  1 , -1 , -1, 0, 0, # v4
        -1 , -1 , -1 , -1, 0, 0, # v7
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

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid():
    # prepare vertex data (in main memory)
    total_vertex = 12*2*401
    vertices = glm.array.zeros(total_vertex, glm.float32)

    # -200 ~ 200 grid
    for i in range(401):
        x = 12*i
        z = 401*12 + x
        vertices[x] = glm.float32(-200)
        vertices[x + 2] = glm.float32(i - 200)
        vertices[x + 6] = glm.float32(200)
        vertices[x + 8] = glm.float32(i - 200)

        vertices[z] = glm.float32(i - 200)
        vertices[z + 2] = glm.float32(-200)
        vertices[z + 6] = glm.float32(i - 200)
        vertices[z + 8] = glm.float32(200)

        vertices[x + 3] = glm.float32(0.5)
        vertices[x + 4] = glm.float32(0.5)
        vertices[x + 5] = glm.float32(0.5)
        vertices[x + 9] = glm.float32(0.5)
        vertices[x + 10] = glm.float32(0.5)
        vertices[x + 11] = glm.float32(0.5)

        vertices[z + 3] = glm.float32(0.5)
        vertices[z + 4] = glm.float32(0.5)
        vertices[z + 5] = glm.float32(0.5)
        vertices[z + 9] = glm.float32(0.5)
        vertices[z + 10] = glm.float32(0.5)
        vertices[z + 11] = glm.float32(0.5)

    # set x-axis, z-axis to red and blue color
    x = 12*200
    z = 401*12 + x
    vertices[x + 3] = glm.float32(1)
    vertices[x + 4] = glm.float32(0)
    vertices[x + 5] = glm.float32(0)
    vertices[x + 9] = glm.float32(1)
    vertices[x + 10] = glm.float32(0)
    vertices[x + 11] = glm.float32(0)
    vertices[z + 3] = glm.float32(0)
    vertices[z + 4] = glm.float32(0)
    vertices[z + 5] = glm.float32(1)
    vertices[z + 9] = glm.float32(0)
    vertices[z + 10] = glm.float32(0)
    vertices[z + 11] = glm.float32(1)

    # create and activate VAO (vertex array object)
    # create a vertex array object ID and store it to VAO variable
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    # create a buffer object ID and store it to VBO variable
    VBO = glGenBuffers(1)
    # activate VBO as a vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, VBO)

    # copy vertex data to VBO
    # allocate GPU memory for and copy vertex data to the currently bound vertex buffer
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW)

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO, total_vertex

# draw grid
def draw_grid(vao, MVP, MVP_loc, total_vertex):
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, total_vertex)

# draw bvh file with lines
def draw_bvh_line(vao, MVP, MVP_loc, color, color_loc, total_vertex):
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, total_vertex)

# draw bvh file with boxes
def draw_bvh_box(vao, node, VP, MVP_loc, M_loc, color, color_loc):
    if node.get_parent() is not None:
        M = node.get_parent().get_global_transform() * node.get_shape_transform()
        MVP = VP * M
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
        glUniform3f(color_loc, color.r, color.g, color.b)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, 12*3)
    for child in node.get_children():
        draw_bvh_box(vao, child, VP, MVP_loc, M_loc, color, color_loc)

# rotate the box to 
def box_rotation(vec):
    up_vec = glm.vec3(0,1,0)
    vec = glm.normalize(vec)

    dot_product = glm.dot(up_vec, vec)

    if abs(dot_product - 1) < 1e-6:
        return glm.mat4()
    elif  abs(dot_product - (-1)) < 1e-6:
        return glm.rotate(glm.radians(180), (1,0,0))

    axis = glm.cross(up_vec, vec)
    angle = glm.acos(glm.dot(up_vec, vec))

    return glm.rotate(angle, axis)

def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)

    # create a window and OpenGL context
    window = glfwCreateWindow(
        950, 950, 'OpenGL viewer - 2019092306 Koo Bonjun', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback)
    glfwSetCursorPosCallback(window, cursor_callback)
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetScrollCallback(window, scroll_callback)
    glfwSetDropCallback(window, drop_callback)

    # load shaders
    shader_for_grid = load_shaders(g_vertex_shader_src_grid, g_fragment_shader_src_grid)
    shader_for_line = load_shaders(g_vertex_shader_src_line, g_fragment_shader_src_line)
    shader_for_box = load_shaders(g_vertex_shader_src_box, g_fragment_shader_src_box)

    # get uniform locations
    MVP_loc_grid = glGetUniformLocation(shader_for_grid, 'MVP')
    MVP_loc_line =  glGetUniformLocation(shader_for_line, 'MVP')
    color_loc_line = glGetUniformLocation(shader_for_line, 'color')
    MVP_loc_box = glGetUniformLocation(shader_for_box, 'MVP')
    M_loc_box = glGetUniformLocation(shader_for_box, 'M')
    view_pos_loc_box = glGetUniformLocation(shader_for_box, 'view_pos')
    color_loc_box = glGetUniformLocation(shader_for_box, 'mat_color')

    # prepare vaos
    vao_grid, total_vertex_grid = prepare_vao_grid()
    vao_box = prepare_vao_box()

    # set default color
    default_color = glm.vec3(0.2,0.7,1.0)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        start_time = glfwGetTime()

        # enable depth test
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # projection matrix
        # orthogonal projection mode / perspective projection mode
        if proj_mode == 1:
            P = glm.ortho(-1,1,-1,1,-1,1)
        else:
            P = glm.perspective(45,1,0.1,100)

        # viewing matrix
        # V1: orbit right/left, V2: orbit up/down, V3: pan
        V1 = glm.lookAt(glm.vec3(.1*np.sin(orbit_x),0,.1*np.cos(orbit_x)), glm.vec3(0,0,0), glm.vec3(0,1,0))
        V2 = glm.lookAt(glm.vec3(0,.1*np.sin(orbit_y),.1*np.cos(orbit_y)), glm.vec3(0,0,0), glm.vec3(0,flip,0))
        V3 = glm.lookAt(glm.vec3(pan_x,pan_y,zoom_deg), glm.vec3(pan_x,pan_y,-1), glm.vec3(0,1,0))
        V = V3*V2*V1

        # view point position for using in phong shading
        view_pos = glm.inverse(V)[3]

        # modeling matrix (default)
        M = glm.mat4()

        # current frame: P*V*M (now this is the world frame)
        VP = P*V
        MVP = P*V*M

        # draw current grid
        glUseProgram(shader_for_grid)
        draw_grid(vao_grid, MVP, MVP_loc_grid, total_vertex_grid)

        # draw bvh
        if vao_drop != 0:
            if is_paused == 0:
                hierarchy_drop.set_joint_transform()
                hierarchy_drop.update_tree_global_transform()

            if rendering_mode == 1:
                points, total_vertex = hierarchy_drop.get_points()
                copy_points_data(points, vbo_drop)
                glUseProgram(shader_for_line)
                draw_bvh_line(vao_drop, MVP, MVP_loc_line, default_color, color_loc_line, total_vertex)
            elif rendering_mode  == 2:
                glUseProgram(shader_for_box)
                glUniform3f(view_pos_loc_box, view_pos.x, view_pos.y, view_pos.z)   
                draw_bvh_box(vao_box, hierarchy_drop, VP, MVP_loc_box, M_loc_box, default_color, color_loc_box)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

        while True:
            end_time = glfwGetTime()
            if(end_time - start_time >= Node.frame_time):
                break

    # terminate glfw
    glfwTerminate()


if __name__ == "__main__":
    main()
