from PIL import Image
from functools import reduce
import numpy as np
import time
import numbers

def extract(cond, x):
    if isinstance(x, numbers.Number):
        return x
    else:
        return np.extract(cond, x)

# Funkcje pomocnicze
class vec3():
    # Deklaracja wartości współrzędnych 3D
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)
    # Mnożenie
    def __mul__(self, other):
        return vec3(self.x * other, self.y * other, self.z * other)
    # Dodanie
    def __add__(self, other):
        return vec3(self.x + other.x, self.y + other.y, self.z + other.z)
    # Odejmowanie
    def __sub__(self, other):
        return vec3(self.x - other.x, self.y - other.y, self.z - other.z)
    # Obliczanie macierzy
    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    # Wartość bezwzględna
    def __abs__(self):
        return self.dot(self)
    # Wektor jednostkowy
    def norm(self):
        mag = np.sqrt(abs(self))
        return self * (1.0 / np.where(mag == 0, 1, mag))
    def components(self):
        return (self.x, self.y, self.z)
    def extract(self, cond):
        return vec3(extract(cond, self.x),
                    extract(cond, self.y),
                    extract(cond, self.z))
    # Zmiana parametrów w tablicy
    def place(self, cond):
        r = vec3(np.zeros(cond.shape), np.zeros(cond.shape), np.zeros(cond.shape))
        np.place(r.x, cond, self.x)
        np.place(r.y, cond, self.y)
        np.place(r.z, cond, self.z)
        return r
rgb = vec3

(w, h) = (400, 300)        # Rozmiar obrazu
L = vec3(5, 5, -5)         # Położenie światła
E = vec3(0, 0.35, -1)      # Położenie kamery
FARAWAY = 1.0e39           # an implausibly huge distance

def raytrace(O, D, scene):
    # O jest początkiem promienia, D jest znormalizowanym kierunkiem promienia
    # scena jest listą zadeklarowanych obiektów (kule)
    # Odbicie jest numerem odbicia zaczynającym się w punkcie kamery

    distances = [s.intersect(O, D) for s in scene]
    nearest = reduce(np.minimum, distances)
    color = rgb(0, 0, 0)
    for (s, d) in zip(scene, distances):
        hit = (nearest != FARAWAY) & (d == nearest)
        if np.any(hit):
            dc = extract(hit, d)
            Oc = O.extract(hit)
            Dc = D.extract(hit)
            cc = s.light(Oc, Dc, dc, scene)
            color += cc.place(hit)
    return color

class Sphere:
    def __init__(self, center, r, diffuse, mirror = 0.5):
        self.c = center         #środek kuli
        self.r = r              #średnica
        self.diffuse = diffuse  #kolor rzeczywisty
        self.mirror = mirror    #odbicie

    # przebieg promienia od kamery przez piksel sceny w poszukiwaniu obiektu
    # po znalezieniu zwraca współrzędne
    def intersect(self, O, D):
        b = 2 * D.dot(O - self.c)
        c = abs(self.c) + abs(O) - 2 * self.c.dot(O) - (self.r * self.r)
        disc = (b ** 2) - (4 * c)
        sq = np.sqrt(np.maximum(0, disc))
        h0 = (-b - sq) / 2
        h1 = (-b + sq) / 2
        h = np.where((h0 > 0) & (h0 < h1), h0, h1)
        pred = (disc > 0) & (h > 0)
        return np.where(pred, h, FARAWAY)

    def diffusecolor(self, M):
        return self.diffuse

    def light(self, O, D, d, scene):
        M = (O + D * d)                         # punkt przecięcia z obiektem
        N = (M - self.c) * (1. / self.r)        # znormalizowanie
        toL = (L - M).norm()                    # kierunek do światła
        toO = (E - M).norm()                    # kierunek do kamery
        nudged = M + N * .0001                  # przekierowanie M

        # Sprawdza czy danym punkcie tworzy się cień
        light_distances = [s.intersect(nudged, toL) for s in scene]
        light_nearest = reduce(np.minimum, light_distances)
        seelight = light_distances[scene.index(self)] == light_nearest

        # Ambient
        color = rgb(0.05, 0.05, 0.05)

        # Kolor rzeczywisty
        lv = np.maximum(N.dot(toL), 0)
        color += self.diffusecolor(M) * lv * seelight

        # Obliczanie koloru z modelu Blinn-Phong'a
        phong = N.dot((toL + toO).norm())
        color += rgb(1, 1, 1) * np.power(np.clip(phong, 0, 1), 50) * seelight
        return color

class CheckeredSphere(Sphere):
    def diffusecolor(self, M):
        checker = ((M.x * 2).astype(int) % 2) == ((M.z * 2).astype(int) % 2)
        return self.diffuse * checker

# definicja obiektu i podłogi
scene = [
    Sphere(vec3(.75, .1, 1), .6, rgb(0, 0, 1)),
    Sphere(vec3(-.75, .1, 2.25), .6, rgb(.5, .223, .5)),
    Sphere(vec3(-2.75, .1, 3.5), .6, rgb(1, .572, .184)),
    CheckeredSphere(vec3(0,-99999.5, 0), 99999, rgb(.75, .75, .75), 0.25),
    ]

# Skalowanie proporcji obrazu
r = float(w) / h
# Koordynaty sceny: x0, y0, x1, y1.
S = (-1, 1 / r + .25, 1, -1 / r + .25)
x = np.tile(np.linspace(S[0], S[2], w), h)
y = np.repeat(np.linspace(S[1], S[3], h), w)

t0 = time.time()
Q = vec3(x, y, 0)
color = raytrace(E, (Q - E).norm(), scene)
print ("Took", time.time() - t0)

# Zapisywanie obrazu do pliku

rgb = [Image.fromarray((255 * np.clip(c, 0, 1).reshape((h, w))).astype(np.uint8), "L") for c in color.components()]
Image.merge("RGB", rgb).save("RayTracer.png")