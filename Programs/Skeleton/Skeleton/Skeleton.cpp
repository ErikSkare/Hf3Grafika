//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

GPUProgram gpuProgram;

float randomFloatBetween(float min, float max) {
	float dist = max - min;
	float rnd = (float)rand() / (float)RAND_MAX;
	return min + rnd * dist;
}

float selfMax(float a, float b) {
	if (a > b)
		return a;
	return b;
}

float selfMin(float a, float b) {
	if (a < b)
		return a;
	return b;
}

class Camera {
	float fov, asp, fp, bp;

protected:
	vec3 wEye, wLookAt, wVup;

public:
	Camera(vec3 wEye, vec3 wLookAt, vec3 wVup): wEye(wEye), wLookAt(wLookAt), wVup(wVup) {
		this->fov = 75 * (float)M_PI / 180.0f;
		this->asp = 0.5f;
		this->fp = 1;
		this->bp = 50;
	}
	
	mat4 getViewMatrix() {
		vec3 w = normalize(wEye - wLookAt);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = normalize(cross(w, u));
		return 
			TranslateMatrix(wEye * (-1)) *
			mat4({ u.x, v.x, w.x, 0 },
				 { u.y, v.y, w.y, 0 },
				 { u.z, v.z, w.z, 0 },
				 { 0  , 0  , 0  , 1 });
	}

	mat4 getProjectionMatrix() {
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
					0, 1 / tan(fov / 2), 0, 0,
					0, 0, -(fp + bp) / (bp - fp), -1,
					0, 0, -2 * fp * bp / (bp - fp), 0);
	}

	void setUniforms() {
		gpuProgram.setUniform(getViewMatrix(), "V");
		gpuProgram.setUniform(getProjectionMatrix(), "P");
		gpuProgram.setUniform(wEye, "wEye");
	}
};

class DroneCamera : public Camera {
	float period;
	vec3 startingW;

public:
	DroneCamera(vec3 wEye, vec3 wLookAt, vec3 wVup, float period): Camera(wEye, wLookAt, wVup), period(period) {
		startingW = wEye - wLookAt;
	}

	void animate(float time) {
		vec4 rotatedW = vec4(startingW.x, startingW.y, startingW.z, 1.0f) * RotationMatrix(time * 2 * (float)M_PI / period, vec3(0, 0, 1));
		wEye = wLookAt + vec3(rotatedW.x, rotatedW.y, rotatedW.z);
	}
};

class BodyCamera : public Camera {
public:
	BodyCamera(vec3 center, vec3 normal, vec3 side): Camera(center, center + normal, cross(normal, side)) {}

	void set(vec3 center, vec3 normal, vec3 side) {
		wEye = center;
		wLookAt = center + normal;
		wVup = cross(normal, side);
	}
};

struct DirectionalLight {
	vec3 direction;
	vec3 Lin;

	DirectionalLight(vec3 direction, vec3 Lin): direction(direction), Lin(Lin) {}

	void setUniforms() {
		gpuProgram.setUniform(direction, "wLdir");
		gpuProgram.setUniform(Lin, "wLin");
	}
};

struct Material {
	vec3 kdFrom, kdTo, ks;
	float shininess;

	void setUniforms() {
		gpuProgram.setUniform(kdFrom, "kdFrom");
		gpuProgram.setUniform(kdTo, "kdTo");
		gpuProgram.setUniform(ks, "ks");
		gpuProgram.setUniform(shininess, "shininess");
	}
};

class Body {
	struct VertexData {
		vec3 position, normal;
	};

	Material material;
	float a, b, c;

	vec3 center, axis; vec3 phi = { 0, 0, 0 };
	float m; float I;
	vec3 p; vec3 L;
	vec3 v = { 0, 0, 0 }; vec3 w = { 0, 0, 0 };
	
	vec3 g = { 0, 0, -9.81 };
	vec3 s; float l0; float D;

	float rho = 10, ki = 10;

	bool isReady = false;

	unsigned int vao;

public:
	Body(vec3 center, float a, float b, float c, float m, float l0, float D): center(center), a(a), b(b), c(c), m(m), l0(l0), D(D) {
		this->vao = NULL;
		this->axis = vec3(0, 1, 0);
		this->phi = 0;
		this->I = m * (a * a + c * c) / 12.0f;
		this->s = center + vec3(0, 0, -c / 2);
		this->material.kdFrom = vec3(210 / 255.0f, 105.0f / 255.0f, 30.0f / 255.0f);
		this->material.kdTo = vec3(210 / 255.0f, 105.0f / 255.0f, 30.0f / 255.0f);
		this->material.ks = vec3(0, 0, 0);
		this->material.shininess = 1;
	}

	void draw(Camera& camera) {
		glBindVertexArray(vao);
		material.setUniforms();
		camera.setUniforms();
		gpuProgram.setUniform(getModelMatrix(), "M");
		gpuProgram.setUniform(getInverseModelMatrix(), "MInv");
		gpuProgram.setUniform(-c / 2, "minZ");
		gpuProgram.setUniform(c / 2, "maxZ");
		for (int i = 0; i < 6; ++i) {
			glDrawArrays(GL_TRIANGLE_STRIP, i * 4, 4);
		}
	}

	mat4 getModelMatrix() {
		return RotationMatrix(phi.x, { 1, 0, 0 }) * RotationMatrix(phi.y, { 0, 1, 0 }) * RotationMatrix(phi.z, { 0, 0, 1 }) * TranslateMatrix(center);
	}

	mat4 getInverseModelMatrix() {
		return TranslateMatrix(-center) * RotationMatrix(-phi.x, { 1, 0, 0 }) * RotationMatrix(-phi.y, { 0, 1, 0 }) * RotationMatrix(-phi.z, { 0, 0, 1 });
	}

	void setReady() {
		if (isReady) return;
		v = { randomFloatBetween(2, 4), 0, randomFloatBetween(0, 4) };
		p = m * v;
		isReady = true;
	}

	void step(float dt) {
		if (!isReady) return;
		vec3 l = transformPoint3D(vec3(0, 0, -c / 2));

		vec3 K(0, 0, 0);
		if (length(s - l) > l0) {
			K = D * normalize(s - l) * (length(s - l) - l0);
		}

		vec3 F = m * g + K - rho * v;
		vec3 M = cross((l - center), K) - ki * w;

		p = p + F * dt;
		L = L + M * dt;

		v = p / m;
		w = L / I;
		
		center = center + v * dt;
		phi = phi + w * dt;
	}
	
	void create() {
		std::vector<VertexData> vertices;
		for (int i = -1; i <= 1; i += 2) {
			vertices.push_back({ vec3(i * a / 2, b / 2, c / 2), vec3(i, 0, 0) });
			vertices.push_back({ vec3(i * a / 2, b / 2, -c / 2), vec3(i, 0, 0) });
			vertices.push_back({ vec3(i * a / 2, -b / 2, c / 2), vec3(i, 0, 0) });
			vertices.push_back({ vec3(i * a / 2, -b / 2, -c / 2), vec3(i, 0, 0) });
		}
		for (int i = -1; i <= 1; i += 2) {
			vertices.push_back({ vec3(a / 2, i * b / 2, c / 2), vec3(0, i, 0) });
			vertices.push_back({ vec3(a / 2, i * b / 2, -c / 2), vec3(0, i, 0) });
			vertices.push_back({ vec3(-a / 2, i * b / 2, c / 2), vec3(0, i, 0) });
			vertices.push_back({ vec3(-a / 2, i * b / 2, -c / 2), vec3(0, i, 0) });
		}
		for (int i = -1; i <= 1; i += 2) {
			vertices.push_back({ vec3(a / 2, b / 2, i * c / 2), vec3(0, 0, i) });
			vertices.push_back({ vec3(a / 2, -b / 2, i * c / 2), vec3(0, 0, i) });
			vertices.push_back({ vec3(-a / 2, b / 2, i * c / 2), vec3(0, 0, i) });
			vertices.push_back({ vec3(-a / 2, -b / 2, i * c / 2), vec3(0, 0, i) });
		}
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}

	vec3 getSideA() {
		vec3 p1 = transformPoint3D(vec3(0, -b / 2, c / 2));
		vec3 p2 = transformPoint3D(vec3(0, b / 2, c / 2));
		return p2 - p1;
	}

	vec3 getSideACenter() {
		return transformPoint3D(vec3(0, 0, c / 2));
	}

	vec3 getSideANormal() {
		return getSideACenter() - center;
	}

private:
	vec3 transformPoint3D(vec3 p) {
		vec4 a = vec4(p.x, p.y, p.z, 1);
		vec4 result = a * getModelMatrix();
		return vec3(result.x, result.y, result.z);
	}
};

class Terrain {
	Material material;

	struct VertexData {
		vec3 position, normal;
	};

	int n;
	float a0;
	float size;
	int resolution;
	float** phases;

	float minZ = (float)INT_MAX, maxZ = (float)-INT_MAX;

	unsigned int vao;

public:
	Terrain(int n, float a0, float size, int resolution): n(n), a0(a0), size(size), resolution(resolution) {
		phases = new float*[n+1];
		for (int i = 0; i <= n; ++i) {
			phases[i] = new float[n+1];
			for (int j = 0; j <= n; ++j) {
				phases[i][j] = randomFloatBetween(0, 2 * M_PI);
			}
		}
		material.kdFrom = { 69.0f / 255.0f, 252.0f / 255.0f, 3.0f / 255.0f };
		material.kdTo = { 148.0f / 255.0f, 94.0f / 255.0f, 58.0f / 255.0f };
		material.ks = { 0.2f, 0.1f, 0.1f };
		material.shininess = 5;
	}

	void create() {
		std::vector<VertexData> vertices;
		float step = size / (resolution - 1);
		for (int i = 0; i < resolution - 1; ++i) {
			for (int j = 0; j <= resolution - 1; ++j) {
				float y = -size / 2 + i * step;
				float x = -size / 2 + j * step;
				vertices.push_back({ vec3(x, y, getHeightAt(x, y)), getNormalAt(x, y) });
				vertices.push_back({ vec3(x, y + step, getHeightAt(x, y + step)), getNormalAt(x, y + step) });
			}
		}
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(VertexData) * vertices.size(), &vertices[0], GL_STATIC_DRAW);
		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
	}

	void draw(Camera& camera) {
		mat4 identity = { {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
		glBindVertexArray(vao);
		camera.setUniforms();
		material.setUniforms();
		gpuProgram.setUniform(identity, "M");
		gpuProgram.setUniform(identity, "MInv");
		gpuProgram.setUniform(minZ, "minZ");
		gpuProgram.setUniform(maxZ, "maxZ");
		for (int i = 0; i < resolution - 1; ++i) {
			glDrawArrays(GL_TRIANGLE_STRIP, 2 * i * resolution, 2 * resolution);
		}
	}

	~Terrain() {
		for (int i = 0; i < n; ++i)
			delete[] phases[i];
		delete[] phases;
	}

private:
	float getHeightAt(float x, float y) {
		float value = 0;
		for (int f1 = 0; f1 <= n; ++f1) {
			for (int f2 = 0; f2 <= n; ++f2) {
				if (f1 == 0 && f2 == 0) continue;
				value += a0 / sqrtf(f1 * f1 + f2 * f2) * cosf(f1 * x + f2 * y + phases[f1][f2]);
			}
		}
		maxZ = selfMax(maxZ, value);
		minZ = selfMin(minZ, value);
		return value;
	}

	vec3 getNormalAt(float x, float y) {
		float hx = 0, hy = 0;
		for (int f1 = 0; f1 <= n; ++f1) {
			for (int f2 = 0; f2 <= n; ++f2) {
				if (f1 == 0 && f2 == 0) continue;
				hx += -a0 / sqrtf(f1 * f1 + f2 * f2) * f1 * sinf(f1 * x + f2 * y + phases[f1][f2]);
				hy += -a0 / sqrtf(f1 * f1 + f2 * f2) * f2 * sinf(f1 * x + f2 * y + phases[f1][f2]);
			}
		}
		return vec3(-hx, -hy, 1);
	}
};

const char * const vertexSource = R"(
	#version 330
	precision highp float;

	uniform mat4 M;
	uniform mat4 MInv;
	uniform mat4 V;
	uniform mat4 P;

	uniform vec3 kdFrom;
	uniform vec3 kdTo;
	uniform float minZ;
	uniform float maxZ;
	uniform vec3 ks;
	uniform float shininess;

	uniform vec3 wLdir;
	uniform vec3 wLin;
	uniform vec3 wEye;
	
	layout(location = 0) in vec3 vp;
	layout(location = 1) in vec3 norm;

	out vec3 color;

	void main() {
		gl_Position = vec4(vp, 1) * M * V * P;

		vec4 wPos = vec4(vp, 1) * M;
		vec4 wNormal = MInv * vec4(norm, 0);

		vec3 L = normalize(-wLdir);
		vec3 V = normalize(wEye * wPos.z - wPos.xyz);
		vec3 N = normalize(wNormal.xyz);
		vec3 H = normalize(L + V);

		float cost = max(dot(N, L), 0), cosd = max(dot(N, H), 0);

		color = ((kdFrom + (vp.z - minZ) / (maxZ - minZ) * (kdTo - kdFrom)) * cost + ks * pow(cosd, shininess)) * wLin;
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	in vec3 color;
	out vec4 outColor;

	void main() {
		outColor = vec4(color, 1);
	}
)";

DirectionalLight dLight(vec3(-1, -1, -3), vec3(1.0f, 165.0f / 255.0f, 0.0f ));
Body body(vec3(0, 0, 12), 0.25, 0.5, 1.85, 80, 2, 400);
Terrain terrain(2, 1, 20, 200);
DroneCamera dCamera(vec3(0, 6, 14), vec3(0, 0, 10), vec3(0, 0, 1), 10);
BodyCamera bCamera(body.getSideACenter(), body.getSideANormal(), body.getSideA());

void onInitialization() {
	gpuProgram.create(vertexSource, fragmentSource, "outColor");
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	body.create();
	terrain.create();
	bCamera.set(body.getSideACenter(), body.getSideACenter(), body.getSideA());
}

void onDisplay() {
	glClearColor(128 / 255.0f, 127 / 255.0f, 203 / 255.0f, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glViewport(300, 0, 300, 600);
	dLight.setUniforms();
	body.draw(dCamera);
	terrain.draw(dCamera);

	bCamera.set(body.getSideACenter(), body.getSideANormal(), body.getSideA());
	glViewport(0, 0, 300, 600);
	body.draw(bCamera);
	terrain.draw(bCamera);

	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	body.setReady();
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

float before = 0;
float step = 0.005f;

void onIdle() {
	float now = (float)glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	float dt = now - before;
	before = now;

	dCamera.animate(now);
	for (float s = 0; s <= dt; s += step)
		body.step(selfMin(step, dt - s));

	glutPostRedisplay();
}
