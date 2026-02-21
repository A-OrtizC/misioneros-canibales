import cv2
import numpy as np
import mss
import pyautogui
import time

estados_terminar = [((0, 0, 0, 0), (0, 0, 0, 0)), ((3, 0, 0, 0), (3, 0, 0, 0))]

class Agent:
    B1 = ((0,3,0,0),(0,3,0,0))
    B2 = ((0,2,0,1),(0,3,0,0))
    B3 = ((0,1,0,2),(0,3,0,0))
    B4 = ((0,1,2,0),(0,3,0,0))
    B5 = ((1,1,1,0),(0,3,0,0))
    B6 = ((1,1,0,1),(0,3,0,0))
    B7 = ((1,0,0,2),(0,3,0,0))
    B8 = ((1,0,2,0),(0,3,0,0))
    B9 = ((2,0,1,0),(0,3,0,0))
    B10 = ((2,0,0,1),(0,3,0,0))
    B11 = ((2,1,0,0),(0,3,0,0))
    B12 = ((2,1,0,0),(0,2,0,1))
    B13 = ((2,1,0,0),(0,1,0,2))
    B14 = ((2,1,0,0),(0,1,2,0))
    B15 = ((2,1,0,0),(1,1,1,0))
    B16 = ((1,1,1,0),(1,1,1,0))
    B17 = ((1,1,0,1),(1,1,0,1))
    B18 = ((1,2,0,0),(1,1,0,1))
    B19 = ((1,2,0,0),(1,0,0,2))
    B20 = ((1,2,0,0),(1,0,2,0))
    B21 = ((1,2,0,0),(2,0,1,0))
    B22 = ((1,2,0,0),(3,0,0,0))
    B23 = ((0,2,1,0),(3,0,0,0))
    B24 = ((0,2,0,1),(3,0,0,0))
    B25 = ((0,1,0,2),(3,0,0,0))
    B26 = ((0,1,2,0),(3,0,0,0))
    B27 = ((1,1,1,0),(3,0,0,0))
    B28 = ((1,1,0,1),(3,0,0,0))
    B29 = ((1,0,0,2),(3,0,0,0))
    B30 = ((1,0,2,0),(3,0,0,0))
    B31 = ((2,0,1,0),(3,0,0,0))
    B32 = ((3,0,0,0),(3,0,0,0))

    # estados adicionales
    B33 = ((0,2,0,1),(0,2,0,1))
    B34 = ((0,2,1,0),(0,2,1,0))
    B35 = ((1,2,0,0),(0,2,1,0))
    B36 = ((1,2,0,0),(0,2,0,1))
    B37 = ((1,2,0,0),(0,3,0,0))

    # ---------------- GRAFO ----------------
    estados_principales = [
        B1,B2,B3,B4,B5,B6,B7,B8,
        B9,B10,B11,B12,B13,B14,B15,B16,
        B17,B18,B19,B20,B21,B22,B23,B24,
        B25,B26,B27,B28,B29,B30,B31,B32
    ]

    acciones_principales = [
        "CS","CS","K","CB","K","CS","K","CB","K","CB",
        "MS","MS","K","MB","CS","K","CB","MS","K","MB",
        "MB","CS","K","CS","K","CB","K","CS","K","CB","CB"
    ]

    # camino alterno
    estados_extra = [B33,B34,B35,B36,B37,B6]
    acciones_extra = ["K","CB","K","MB","CS"]

    # ---------- CLICK ----------
    def click_en(self, coord, duracion=0.1):
        if coord is None:
            return

        x, y = coord

        pyautogui.moveTo(x, y, duration=duracion,
                        tween=pyautogui.easeInOutQuad)

        pyautogui.click()

    # ---------- ESTADO ----------
    def get_estado(self, p):
        return (
            (min(len(p["cizq"]), 3),
            min(len(p["cder"]), 3),
            min(len(p["cizqb"]), 3),
            min(len(p["cderb"]), 3)),

            (min(len(p["mizq"]), 3),
            min(len(p["mder"]), 3),
            min(len(p["mizqb"]), 3),
            min(len(p["mderb"]), 3))
        )

    # ---------- ACCIONES ----------
    def procesar_accion(self, accion, posiciones):
        der = posiciones["rio"] < posiciones["bote"][0][0]

        if accion == "CS":
            self.click_en(posiciones["cder"][0] if der else posiciones["cizq"][0])

        elif accion == "MS":
            self.click_en(posiciones["mder"][0] if der else posiciones["mizq"][0])

        elif accion == "CB":
            self.click_en(posiciones["cderb"][0] if der else posiciones["cizqb"][0])

        elif accion == "MB":
            self.click_en(posiciones["mderb"][0] if der else posiciones["mizqb"][0])

        elif accion == "K":
            self.click_en(posiciones["bote"][0])
    
    def compute(self, percept):
        estado = self.get_estado(percept)

        accion = None

        for i in range(len(self.estados_principales)-1):
            if estado == self.estados_principales[i]:
                accion = self.acciones_principales[i]
                self.procesar_accion(accion, percept)
                break

        if accion is None:
            for i in range(len(self.estados_extra)-1):
                if estado == self.estados_extra[i]:
                    accion = self.acciones_extra[i]
                    self.procesar_accion(accion, percept)
                    break

        return estado

class Environment:
    sct = mss.mss()

    HEX_CANIBAL = "#c93636"
    HEX_MISIONERO = "#ffb29d"
    HEX_BOTE = "#614a37"
    HEX_RIO = "#6098ad"

    TOL = 12
    DISTANCIA_CLUSTER = 60

    # ---------- HEX → BGR ----------
    def hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return np.array([b, g, r], dtype=np.int16)

    # ---------- CREAR MASK ----------
    def crear_mask(self, img, color):
        lower = np.clip(color - self.TOL, 0, 255).astype(np.uint8)
        upper = np.clip(color + self.TOL, 0, 255).astype(np.uint8)

        return cv2.inRange(img, lower, upper)

    # ---------- CLUSTER RAPIDO ----------
    def clusterizar(self, mask):
        puntos = np.column_stack(np.where(mask > 0))

        if len(puntos) == 0:
            return []

        centros = []

        for y, x in puntos:

            agregado = False

            for i, (cx, cy, count) in enumerate(centros):

                if abs(cx - x) < self.DISTANCIA_CLUSTER and abs(cy - y) < self.DISTANCIA_CLUSTER:

                    centros[i] = (
                        (cx * count + x) / (count + 1),
                        (cy * count + y) / (count + 1),
                        count + 1,
                    )
                    agregado = True
                    break

            if not agregado:
                centros.append((x, y, 1))

        return [(int(cx), int(cy)) for cx, cy, c in centros if c > 200]

    # ---------- DETECTAR RIO ----------
    def detectar_rio(self, mask):
        puntos = np.column_stack(np.where(mask > 0))

        if len(puntos) == 0:
            return 0

        return int(np.median(puntos[:, 1]))
    
    def detectar_elementos(self):
        screen = np.array(self.sct.grab(self.sct.monitors[0]))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)

        mask_canibal = self.crear_mask(screen, self.COLOR_CANIBAL)
        mask_misionero = self.crear_mask(screen, self.COLOR_MISIONERO)
        mask_bote = self.crear_mask(screen, self.COLOR_BOTE)
        mask_rio = self.crear_mask(screen, self.COLOR_RIO)

        canibales = self.clusterizar(mask_canibal)
        misioneros = self.clusterizar(mask_misionero)
        botes = self.clusterizar(mask_bote)

        x_rio = self.detectar_rio(mask_rio)

        posiciones = {
            "cizq": [], "cder": [],
            "mizq": [], "mder": [],
            "cizqb": [], "cderb": [],
            "mizqb": [], "mderb": [],
            "bote": []
        }

        bote = botes[0] if botes else None

        if bote:
            posiciones["bote"].append(bote)

        margen_botex = 200
        margen_botey = 100
        margen_boteycan = 250

        def esta_en_bote(canibal, x, y):

            if bote is None:
                return False

            bx, by = bote

            if canibal:
                return abs(x - bx) < margen_botex and abs(y - by) < margen_boteycan

            return abs(x - bx) < margen_botex and abs(y - by) < margen_botey

        # CANIBALES
        for x, y in canibales:

            if esta_en_bote(True, x, y):
                (posiciones["cizqb"] if x < x_rio else posiciones["cderb"]).append((x, y))
            else:
                (posiciones["cizq"] if x < x_rio else posiciones["cder"]).append((x, y))

        # MISIONEROS
        for x, y in misioneros:

            if esta_en_bote(False, x, y):
                (posiciones["mizqb"] if x < x_rio else posiciones["mderb"]).append((x, y))
            else:
                (posiciones["mizq"] if x < x_rio else posiciones["mder"]).append((x, y))

        posiciones["rio"] = x_rio

        return posiciones

    def __init__(self):
        self.COLOR_CANIBAL = self.hex_to_bgr(self.HEX_CANIBAL)
        self.COLOR_MISIONERO = self.hex_to_bgr(self.HEX_MISIONERO)
        self.COLOR_BOTE = self.hex_to_bgr(self.HEX_BOTE)
        self.COLOR_RIO = self.hex_to_bgr(self.HEX_RIO)

    def getPercept(self):
        return self.detectar_elementos()

time.sleep(2)

#Main program
agent = Agent()
env = Environment()

estado = ()

while estado not in estados_terminar:
    percept = env.getPercept()
    estado = agent.compute(percept)