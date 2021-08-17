/*
 * Copyright (C) 2021, Kirill GPRB.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <assert.h>
#include <errno.h>
#include <float.h>
#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION 1
#include "stb_image_write.h"

#define MACROSTR1(x) #x
#define MACROSTR2(x) MACROSTR1(x)

#define WIDTH   (1152)
#define HEIGHT  (648)
#define COLOR_R (0)
#define COLOR_G (255)
#define COLOR_B (0)

typedef float vec2_t[2];

struct graphdata_s {
    /* tags */
    int msaa;
    int save;
    float line_width;
    float frame_px;

    /* calculated */    
    float max_value;
    float min_value;
    float tick_size;
    size_t size;
    float *data;
};

static struct graphdata_s graphdata = { 0 };
static GLFWwindow *window = NULL;
static GLuint glprogram = 0;
static GLuint glvao = 0;
static GLuint glvbo = 0;

static const char *glsl_v =
    "#version 450\n"
    "const int WIDTH = " MACROSTR2(WIDTH) ";\n"
    "const int HEIGHT = " MACROSTR2(HEIGHT) ";\n"
    "layout(location = 0) in vec2 position;"
    "void main(void)\n"
    "{\n"
    "gl_Position = vec4(vec2(position.x / WIDTH, position.y / HEIGHT) * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n";

static const char *glsl_f =
    "#version 450\n"
    "layout(location = 0) out vec4 target;\n"
    "void main(void)\n"
    "{\n"
    "target = vec4(vec3(" MACROSTR2(COLOR_R) ", " MACROSTR2(COLOR_G) ", " MACROSTR2(COLOR_B) ") / 255.0, 1.0);\n"
    "}\n";

static void lprintf(const char *fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    vfprintf(stderr, fmt, va);
    va_end(va);
}

static void on_glfw_error(int code, const char *message)
{
    lprintf("GLFW error %d: %s\n", code, message);
}

static int read_undgraph(const char *filename, struct graphdata_s *data)
{
    int nc, nr;
    float f;
    size_t i;
    char line[64], tag[32];
    const char *lp;
    FILE *fp;
    
    fp = fopen(filename, "r");
    if(!fp) {
        lprintf("%s\n", strerror(errno));
        return 0;
    }

    /* default tag values */
    data->msaa = 0;
    data->save = 0;
    data->line_width = 1.0f;
    data->frame_px = 0;

    /* header */
    nc = 0;
    lp = fgets(line, sizeof(line), fp);
    i = strlen(lp);

    /* header: magic */
    sscanf(lp += nc, "%31s %n", tag, &nc);
    if(strcmp(tag, "undgraph")) {
        lprintf("%s: invalid header format\n");
        goto error;
    }

    /* header: tags */
    while((size_t)(lp - line) < i && (nr = sscanf(lp += nc, " %31s %n", tag, &nc)) > 0) {
        if(strstr(tag, "msaa") == tag) {
            sscanf(tag, "msaa:%d", &data->msaa);
            continue;
        }

        if(strstr(tag, "save") == tag) {
            sscanf(tag, "save:%d", &data->save);
            continue;
        }

        if(strstr(tag, "lw") == tag) {
            sscanf(tag, "lw:%f", &data->line_width);
            continue;
        }

        if(strstr(tag, "frame_px") == tag) {
            sscanf(tag, "frame_px:%f", &data->frame_px);
            continue;
        }

        lprintf("%s: warning: unknown tag: %s\n", tag);
    }

    /* line count */
    data->size = 0;
    while(fgets(line, sizeof(line), fp)) {
        if(sscanf(line, "%f", &f) != 1)
            break;
        data->size++;
    }

    lprintf("%s: found %zu values\n", filename, data->size);
    fseek(fp, 0, SEEK_SET);

    /* allocate */
    data->data = malloc(sizeof(float) * data->size);
    assert(("Out of memory!", data->data));

    /* skip the header */
    fgets(line, sizeof(line), fp);

    /* read the graph data */
    i = 0;
    f = 0.0f;
    data->max_value = FLT_MIN;
    data->min_value = FLT_MAX;
    while(fgets(line, sizeof(line), fp) && i < data->size) {
        f = strtof(line, NULL);
        if(f > data->max_value)
            data->max_value = f;
        if(f < data->min_value)
            data->min_value = f;
        data->data[i++] = f;
    }

    data->tick_size = fabsf(data->max_value - data->min_value) / (float)data->size;

    fclose(fp);
    return 1;

error:
    fclose(fp);
    return 0;
}

static GLuint compile_shader(GLenum stage, const char *source)
{
    char *info_log;
    GLuint shader;
    GLint gli;
    
    shader = glCreateShader(stage);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    /* check info log */
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &gli);
    if(gli > 1) {
        info_log = malloc(gli + 1);
        assert(("Out of memory!", info_log));
        info_log[gli] = 0;
        glGetShaderInfoLog(shader, gli, NULL, info_log);
        lprintf("%s\n", info_log);
        free(info_log);
    }

    /* check status */
    glGetShaderiv(shader, GL_COMPILE_STATUS, &gli);
    if(gli != GL_TRUE) {
        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

static GLuint link_program(GLuint vs, GLuint fs)
{
    char *info_log;
    GLuint program;
    GLint gli;

    program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    /* check info log */
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &gli);
    if(gli > 1) {
        info_log = malloc(gli + 1);
        assert(("Out of memory!", info_log));
        info_log[gli] = 0;
        glGetProgramInfoLog(program, gli, NULL, info_log);
        lprintf("%s\n", info_log);
        free(info_log);
    }

    /* check status */
    glGetProgramiv(program, GL_LINK_STATUS, &gli);
    if(gli != GL_TRUE) {
        glDeleteProgram(program);
        return 0;
    }

    return program;
}

static const char *bool_to_string(int value)
{
    if(value)
        return "true";
    return "false";
}

int main(int argc, char **argv)
{
    size_t i;
    vec2_t *mesh;
    GLuint vs, fs;
    char tmpstr[128] = { 0 };
    const char *filename;
    char *pixels;

    if(argc > 1) {
        lprintf("reading %s\n", argv[1]);
        filename = argv[1];
    }
    else {
        lprintf("no undgraph file specified, using default: undgraph.txt\n");
        filename = "undgraph.txt";
    }

    if(!read_undgraph(filename, &graphdata))
        return 1;

    for(i = 2; i < (size_t)argc; i++) {
        if(!strcmp(argv[i], "forcemsaa")) {
            graphdata.msaa = 1;
            continue;
        }
        if(!strcmp(argv[i], "forcesave")) {
            graphdata.save = 1;
            continue;
        }
    }

    lprintf("window: %dx%d\n", WIDTH, HEIGHT);
    lprintf("color: #%02X%02X%02XFF\n", COLOR_R, COLOR_G, COLOR_B);
    lprintf("msaa: %s\n", bool_to_string(graphdata.msaa));
    lprintf("save: %s\n", bool_to_string(graphdata.save));
    lprintf("line_width: %f\n", graphdata.line_width);
    lprintf("frame_px: %f\n", graphdata.frame_px);

    if(graphdata.frame_px <= FLT_EPSILON) {
        /* this can cause the graph to sometimes go off limits */
        lprintf("note: frame_px is close to zero. too bad!\n");
    }

    glfwSetErrorCallback(&on_glfw_error);
    if(!glfwInit())
        return 1;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    glfwWindowHint(GLFW_SAMPLES, graphdata.msaa ? 4 : 0);

    snprintf(tmpstr, sizeof(tmpstr), "UndGraph - %s", filename);
    window = glfwCreateWindow(WIDTH, HEIGHT, tmpstr, NULL, NULL);
    if(!window) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    if(!gladLoadGL(glfwGetProcAddress)) {
        lprintf("gladLoadGL failed\n");
        goto error;
    }

    lprintf("GL_VERSION: %s\n", glGetString(GL_VERSION));

    vs = compile_shader(GL_VERTEX_SHADER, glsl_v);
    fs = compile_shader(GL_FRAGMENT_SHADER, glsl_f);
    if(!vs || !fs) {
        lprintf("shader compilation failed\n");
        goto error;
    }

    glprogram = link_program(vs, fs);
    if(!glprogram) {
        lprintf("program link failed\n");
        goto error;
    }

    glDeleteShader(fs);
    glDeleteShader(vs);

    mesh = malloc(sizeof(vec2_t) * graphdata.size);
    assert(("Out of memory!", mesh));
    for(i = 0; i < graphdata.size; i++) {
        mesh[i][0] = (float)graphdata.frame_px + (float)i * (float)(WIDTH - graphdata.frame_px * 2) / (float)graphdata.size;
        mesh[i][1] = (float)graphdata.frame_px + graphdata.data[i] / graphdata.max_value * (float)(HEIGHT - graphdata.frame_px * 2);
        if(isinf(mesh[i][0]))
            lprintf("warning: vertex[%zu].x = infinity\n", i);
        else if(isnan(mesh[i][0]))
            lprintf("warning: vertex[%zu].x = nan\n", i);
        if(isinf(mesh[i][1]))
            lprintf("warning: vertex[%zu].y = infinity\n", i);
        else if(isnan(mesh[i][1]))
            lprintf("warning: vertex[%zu].y = nan\n", i);
    }

    glCreateBuffers(1, &glvbo);
    glNamedBufferData(glvbo, sizeof(vec2_t) * graphdata.size, mesh, GL_STATIC_DRAW);

    glCreateVertexArrays(1, &glvao);
    glVertexArrayVertexBuffer(glvao, 0, glvbo, 0, sizeof(vec2_t));
    glEnableVertexArrayAttrib(glvao, 0);
    glVertexArrayAttribFormat(glvao, 0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexArrayAttribBinding(glvao, 0, 0);

    glLineWidth(graphdata.line_width);

    while(!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        /* clear */
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        /* draw */
        glBindVertexArray(glvao);
        glUseProgram(glprogram);
        glDrawArrays(GL_LINE_STRIP, 0, graphdata.size);

        /* present */
        glfwSwapBuffers(window);

        /* now while we still need to save, do it */
        if(graphdata.save) {
            graphdata.save = 0;
            pixels = malloc(3 * WIDTH * HEIGHT);
            assert(("Out of memory!", pixels));
            glReadPixels(0, 0, WIDTH, HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, pixels);
            snprintf(tmpstr, sizeof(tmpstr), "%s.png", filename);
            stbi_flip_vertically_on_write(1);
            stbi_write_png(tmpstr, WIDTH, HEIGHT, 3, pixels, 3 * WIDTH);
            free(pixels);
        }
    }

    /* cleanup */
    glDeleteVertexArrays(1, &glvao);
    glDeleteBuffers(1, &glvbo);
    glDeleteProgram(glprogram);

    glfwDestroyWindow(window);
    glfwTerminate();

    free(graphdata.data);

    return 0;

error:
    glfwDestroyWindow(window);
    glfwTerminate();
    return 1;
}
