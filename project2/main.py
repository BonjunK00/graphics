from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import os
import numpy as np

press_left_btn = 0
press_right_btn = 0

prev_xpos = 0
prev_ypos = 0

orbit_x = np.pi/4
orbit_y = np.pi/4
pan_x = 0.
pan_y = 0.
zoom_deg = 3
pers_zoom_deg = 3

proj_mode = 0
solid_mode = 1
anim_mode = 0

flip = 1
orbit_flip = 1

vao_drop = 0
total_vertex_drop = 0

g_vertex_shader_src_phong = '''
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

g_fragment_shader_src_phong = '''
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
    vec3 light_pos = vec3(-2,1.5,1);
    vec3 light_color = vec3(.7,.7,.7);
    vec3 material_color = mat_color;
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = light_color;  // or can be material_color

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;

    // second light source

    // light and properties
    light_pos = vec3(2,1.5,1);
    light_color = vec3(.7,.7,.7);

    // light components
    light_ambient = 0.1*light_color;
    light_diffuse = light_color;
    light_specular = light_color;

    // material components
    material_specular = light_color;  // or can be material_color

    // ambient
    ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    light_dir = normalize(light_pos - surface_pos);

    // diffuse
    diff = max(dot(normal, light_dir), 0);
    diffuse = diff * light_diffuse * material_diffuse;

    // specular
    reflect_dir = reflect(-light_dir, normal);
    spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    specular = spec * light_specular * material_specular;

    color += ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''

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

class Node:
    def __init__(self, parent, scale, color, total_vertex):
        # hierarchy
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        # transform
        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        # shape
        self.scale = scale
        self.color = color

        # total vertex
        self.total_vertex = total_vertex

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self):
        return self.global_transform
    def get_scale(self):
        return self.scale
    def get_color(self):
        return self.color
    def get_total_vertex(self):
        return self.total_vertex

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
    global proj_mode, solid_mode, anim_mode, zoom_deg, pers_zoom_deg
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    else:
        # change the projection mode
        if action==GLFW_PRESS or action==GLFW_REPEAT:
            if key==GLFW_KEY_V:
                if proj_mode == 0:
                    pers_zoom_deg = zoom_deg    # save current zoom degree
                    zoom_deg = 0                # no zoom in orthogonal mode
                else:
                    zoom_deg = pers_zoom_deg    # restore zoom degree
                proj_mode = (proj_mode + 1) % 2
            elif key==GLFW_KEY_Z:
                solid_mode = (solid_mode + 1) % 2
            elif key==GLFW_KEY_H:
                anim_mode = (anim_mode + 1) % 2
                
                

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

def drop_callback(window, paths):
    global anim_mode, vao_drop, total_vertex_drop
    vao_drop, total_vertex_drop, print_list = prepare_vao_objfile(paths[0])

    # print obj file information
    print("<Obj file information>")
    print("Obj file name: %s" % print_list[0])
    print("Total number of faces: %d" % print_list[1])
    print("Number of faces with 3 vertices: %d" % print_list[2])
    print("Number of faces with 4 vertices: %d" % print_list[3])
    print("Number of faces with more than 4 vertices: %d" % print_list[4])
    print()

    # set to single mesh rendering mode
    anim_mode = 0

def prepare_vao_objfile(path):
    vertex = []
    vertex_normal = []
    face_vertex = []
    face_vertex_normal = []
    total_vertex = 0
    print_list = [path, 0, 0, 0, 0]

    # read a obj file
    with open(path, "r") as f:
        lines = f.readlines()
        for line in lines:
            flag = line.split(" ")[0]
            if flag == "v":
                line = line.replace("v ", "")
                line = line.split()
                v = [float(i) for i in line]                
                vertex.append(v)
            elif flag == "vn":
                line = line.replace("vn ", "")
                line = line.split()
                vn = [float(i) for i in line]
                vertex_normal.append(vn)
            elif flag == "f":
                line = line.replace("f ", "")
                line = line.split()
                total_vertex += 3 * (len(line) - 2)
                f = [i.split("/") for i in line]
                fv = []
                fn = []
                for v in f:
                    fv.append(int(v[0]) - 1)
                    if len(v) == 2:
                        fn.append(int(v[1]) - 1)
                    else:
                        fn.append(int(v[2]) - 1)
                face_vertex.append(fv)
                face_vertex_normal.append(fn)
                
                # update print list
                print_list[1] += 1
                if len(line) == 3:
                    print_list[2] += 1
                elif len(line) == 4:
                    print_list[3] += 1
                else:
                    print_list[4] += 1

    avg_normal = []
    vertex_normal_list = [[] for i in vertex]

    # make the list of vertex normal that vertex has
    for i in range(len(face_vertex)):
        for j in range(len(face_vertex[i])):
            vertex_normal_list[face_vertex[i][j]].append(face_vertex_normal[i][j])

    # remove duplicated vertex normal
    for i in range(len(vertex_normal_list)):
        vertex_normal_list[i] = list(set(vertex_normal_list[i]))
    
    # figure out average vertex normal per vertex
    for v in vertex_normal_list:
        sum_normal = [0, 0, 0]
        for vn in v:
            for i in range(3):
                sum_normal[i] += vertex_normal[vn][i]
        for i in range(3):
            if len(v) != 0:
                sum_normal[i] = sum_normal[i] / len(v)
        avg_normal.append(sum_normal)
    
    # make vertices to glm array
    vertices = glm.array.zeros(3 * (len(vertex) + len(avg_normal)), glm.float32)
    for i in range(len(vertex)):
        for j in range(6):
            if(j < 3):
                vertices[6*i + j] = glm.float32(vertex[i][j])
            else:
                vertices[6*i + j] = glm.float32(avg_normal[i][j - 3])
    
    # make indices to glm array
    indices = glm.array.zeros(total_vertex, glm.uint32)
    start = 0
    for i in range(len(face_vertex)):
        for j in range(len(face_vertex[i]) - 2):
            indices[start + j * 3] = glm.uint32(face_vertex[i][0])
            indices[start + j * 3 + 1] = glm.uint32(face_vertex[i][j + 1])
            indices[start + j * 3 + 2] = glm.uint32(face_vertex[i][j + 2])
        start += 3 * (len(face_vertex[i]) - 2)       

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # create and activate EBO (element buffer object)
    EBO = glGenBuffers(1)   # create a buffer object ID and store it to EBO variable
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)  # activate EBO as an element buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # copy index data to EBO
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy index data to the currently bound element buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex normals
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO, total_vertex, print_list

def prepare_vao_grid():
    # prepare vertex data (in main memory)
    total_vertex = 12*2*401
    vertices = glm.array.zeros(total_vertex, glm.float32)

    # -200 ~ 200 grid
    for i in range(401):
        x = 12*i
        z = 401*12 + x
        vertices[x] = glm.float32(-20)
        vertices[x + 2] = glm.float32(.1*(i - 200))
        vertices[x + 6] = glm.float32(20)
        vertices[x + 8] = glm.float32(.1*(i - 200))

        vertices[z] = glm.float32(.1*(i - 200))
        vertices[z + 2] = glm.float32(-20)
        vertices[z + 6] = glm.float32(.1*(i - 200))
        vertices[z + 8] = glm.float32(20)

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

    return VAO, total_vertex

# draw grid
def draw_grid(vao, MVP, MVP_loc, total_vertex):
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glBindVertexArray(vao)
    glDrawArrays(GL_LINES, 0, total_vertex)      

# draw obj file
def draw_obj(vao, MVP, MVP_loc, color, color_loc, total_vertex):
    glUniformMatrix4fv(MVP_loc, 1, GL_FALSE, glm.value_ptr(MVP))
    glUniform3f(color_loc, color.r, color.g, color.b)
    glBindVertexArray(vao)
    glDrawElements(GL_TRIANGLES, total_vertex, GL_UNSIGNED_INT, None)

# draw node phong
def draw_node(vao, node, VP, MVP_loc, M_loc, color_loc):
    M = node.get_global_transform() * glm.scale(node.get_scale())
    MVP = VP * M
    color = node.get_color()
    total_vertex = node.get_total_vertex()
    glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
    draw_obj(vao, MVP, MVP_loc, color, color_loc, total_vertex)

def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls

    # create a window and OpenGL context
    window = glfwCreateWindow(950, 950, 'OpenGL viewer - 2019092306 Koo Bonjun', None, None)
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
    shader_for_grid = load_shaders(g_vertex_shader_src, g_fragment_shader_src)
    shader_for_obj = load_shaders(g_vertex_shader_src_phong, g_fragment_shader_src_phong)

    # get uniform locations
    MVP_loc_grid = glGetUniformLocation(shader_for_grid, 'MVP')
    MVP_loc = glGetUniformLocation(shader_for_obj, 'MVP')
    M_loc = glGetUniformLocation(shader_for_obj, 'M')
    view_pos_loc = glGetUniformLocation(shader_for_obj, 'view_pos')
    color_loc = glGetUniformLocation(shader_for_obj, 'mat_color')
    
    # prepare vaos
    vao_grid, total_vertex_grid = prepare_vao_grid()
    vao_horse, total_vertex_holse, print_list_horse = prepare_vao_objfile(os.path.join("horse.obj"))
    vao_bird, total_vertex_bird, print_list_bird = prepare_vao_objfile(os.path.join("bird.obj"))
    vao_pig, total_vertex_pig, print_list_pig = prepare_vao_objfile(os.path.join("pig.obj"))

    # create a hirarchical model 
    horse = Node(None, glm.vec3(.15,.15,.15), glm.vec3(0,1,0), total_vertex_holse)
    bird1 = Node(horse, glm.vec3(.015,.015,.015), glm.vec3(0,0,1), total_vertex_bird)
    bird2 = Node(horse, glm.vec3(.015,.015,.015), glm.vec3(0,0,1), total_vertex_bird)
    pig1 = Node(bird1, glm.vec3(.2,.2,.2), glm.vec3(1,0,0), total_vertex_pig)
    pig2 = Node(bird1, glm.vec3(.2,.2,.2), glm.vec3(1,0,0), total_vertex_pig)
    pig3 = Node(bird2, glm.vec3(.2,.2,.2), glm.vec3(1,0,0), total_vertex_pig)
    pig4 = Node(bird2, glm.vec3(.2,.2,.2), glm.vec3(1,0,0), total_vertex_pig)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # enable depth test
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        # toggle wireframe mode / solid mode
        if solid_mode == 1:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

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

        # set default color for using in single mesh mode
        default_color = glm.vec3(0.2,0.7,1.0)

        # prepare for drawing obj
        glUseProgram(shader_for_obj)
        glUniformMatrix4fv(M_loc, 1, GL_FALSE, glm.value_ptr(M))
        glUniform3f(view_pos_loc, view_pos.x, view_pos.y, view_pos.z)            

        # draw obj
        if vao_drop != 0 and anim_mode == 0:
            # draw single mesh obj
            draw_obj(vao_drop, MVP, MVP_loc, default_color, color_loc, total_vertex_drop)
        elif anim_mode == 1:
             # set local transformations of each node
            t = glfwGetTime()
            horse.set_transform(glm.translate(glm.vec3(0,0,glm.sin(t))))
            bird1.set_transform(glm.translate(glm.vec3(0,.2*glm.sin(3.5*t + np.pi),0))*glm.translate(glm.vec3(0,1.5,.2)))
            bird2.set_transform(glm.translate(glm.vec3(0,.2*glm.sin(3.5*t),0))*glm.translate(glm.vec3(0,1.5,-0.3)))
            pig1.set_transform(glm.translate(glm.vec3(0.2,0,0))*glm.rotate(10*t, glm.vec3(0,1,0))*glm.translate(glm.vec3(-.25,.25,-.10)))
            pig2.set_transform(glm.translate(glm.vec3(-0.2,0,0))*glm.rotate(-10*t, glm.vec3(0,1,0))*glm.translate(glm.vec3(-.25,.25,-.10)))
            pig3.set_transform(glm.translate(glm.vec3(0.2,0,0))*glm.rotate(10*t, glm.vec3(0,1,0))*glm.translate(glm.vec3(-.25,.25,-.10)))
            pig4.set_transform(glm.translate(glm.vec3(-0.2,0,0))*glm.rotate(-10*t, glm.vec3(0,1,0))*glm.translate(glm.vec3(-.25,.25,-.10)))

            # recursively update global transformations of all nodes
            horse.update_tree_global_transform()

            #draw hierarchical model
            draw_node(vao_horse, horse, VP, MVP_loc, M_loc, color_loc)
            draw_node(vao_bird, bird1, VP, MVP_loc, M_loc, color_loc)
            draw_node(vao_bird, bird2, VP, MVP_loc, M_loc, color_loc)
            draw_node(vao_pig, pig1, VP, MVP_loc, M_loc, color_loc)
            draw_node(vao_pig, pig2, VP, MVP_loc, M_loc, color_loc)
            draw_node(vao_pig, pig3, VP, MVP_loc, M_loc, color_loc)
            draw_node(vao_pig, pig4, VP, MVP_loc, M_loc, color_loc)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
