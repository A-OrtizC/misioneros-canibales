import cv2
import numpy as np
import mss
import pyautogui
import time
from collections import deque

estados_terminar = [((0, 0, 0, 0), (0, 0, 0, 0)), ((3, 0, 0, 0), (3, 0, 0, 0))]

class Agent:
    
    MOVES = [(1,0), (2,0), (0,1), (0,2), (1,1)]
    GOAL = (3, 3, 0)

    def is_valid(self, state):
        Ml, Cl, boat = state
        Mr = 3 - Ml
        Cr = 3 - Cl

        if not (0 <= Ml <= 3 and 0 <= Cl <= 3):
            return False

        if Ml > 0 and Cl > Ml:
            return False
        if Mr > 0 and Cr > Mr:
            return False

        return True

    def successors(self, state):
        Ml, Cl, boat = state
        succs = []

        if boat == 0: 
            sign = -1
        else:       
            sign = 1

        for dm, dc in self.MOVES:
            new_state = (Ml + sign*dm, Cl + sign*dc, 1 - boat)
            if self.is_valid(new_state):
                succs.append((new_state, (dm, dc)))

        return succs

    def bfs(self, start, goal):
        queue = deque([start])
        parent = {start: None}
        move_used = {start: None}

        while queue:
            state = queue.popleft()
            if state == goal:
                return parent, move_used

            for new_state, move in self.successors(state):
                if new_state not in parent:
                    parent[new_state] = state
                    move_used[new_state] = move
                    queue.append(new_state)

        return None, None

    def reconstruct_path(self, parent, move_used, goal):
        path = []
        cur = goal
        while cur is not None:
            path.append((cur, move_used[cur]))
            cur = parent[cur]
        path.reverse()
        return path

    def __init__(self):
        self.remaining_m = 0
        self.remaining_c = 0
        self.is_boarding = False

    def click_en(self, coord, duracion=0.1):
        if coord is None:
            return

        x, y = coord

        pyautogui.moveTo(x, y, duration=duracion,
                         tween=pyautogui.easeInOutQuad)
        pyautogui.click()

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
        print(percept)
        accion = None

        der = percept["rio"] < percept["bote"][0][0]
        boat = 1 if der else 0

        c_on_boat = len(percept["cderb"]) if der else len(percept["cizqb"])
        m_on_boat = len(percept["mderb"]) if der else len(percept["mizqb"])

        if (c_on_boat + m_on_boat > 0) and not self.is_boarding:
            if c_on_boat > 0:
                accion = "CB"
            else:
                accion = "MB"
            self.procesar_accion(accion, percept)
        else:
            if self.remaining_c + self.remaining_m > 0:
                if self.remaining_c > 0:
                    accion = "CS"
                    self.remaining_c -= 1
                else:
                    accion = "MS"
                    self.remaining_m -= 1
                self.procesar_accion(accion, percept)
            elif self.is_boarding:
                accion = "K"
                self.procesar_accion(accion, percept)
                self.is_boarding = False
            else:
                Ml = len(percept["mizq"])
                Cl = len(percept["cizq"])
                current_state = (Ml, Cl, boat)
                if current_state == self.GOAL:
                    return self.get_estado(percept)
                parent, move_used = self.bfs(current_state, self.GOAL)
                if parent is None:
                    print("No se encontró solución.")
                    return self.get_estado(percept)
                path = self.reconstruct_path(parent, move_used, self.GOAL)
                next_move = path[1][1]
                self.remaining_m = next_move[0]
                self.remaining_c = next_move[1]
                self.is_boarding = True

        return self.get_estado(percept)

class Environment:
    sct = mss.mss()

    HEX_CANIBAL = "#c93636"
    HEX_MISIONERO = "#ffb29d"
    HEX_BOTE = "#614a37"
    HEX_RIO = "#6098ad"

    TOL = 12
    DISTANCIA_CLUSTER = 60

    def hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return np.array([b, g, r], dtype=np.int16)

    def crear_mask(self, img, color):
        lower = np.clip(color - self.TOL, 0, 255).astype(np.uint8)
        upper = np.clip(color + self.TOL, 0, 255).astype(np.uint8)

        return cv2.inRange(img, lower, upper)

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
            "bote": [],
            "rio": x_rio
        }

        if botes:
            posiciones["bote"].append(botes[0])

        margen_botex = 170
        margen_botey = 170
        margen_boteycan = 200

        def esta_en_bote(canibal, x, y):
            if not posiciones["bote"]:
                return False
            bx, by = posiciones["bote"][0]
            if canibal:
                return abs(x - bx) < margen_botex and abs(y - by) < margen_boteycan
            return abs(x - bx) < margen_botex and abs(y - by) < margen_botey

        for x, y in canibales:
            if esta_en_bote(True, x, y):
                if x < x_rio:
                    posiciones["cizqb"].append((x, y))
                else:
                    posiciones["cderb"].append((x, y))
            else:
                if x < x_rio:
                    posiciones["cizq"].append((x, y))
                else:
                    posiciones["cder"].append((x, y))

        for x, y in misioneros:
            if esta_en_bote(False, x, y):
                if x < x_rio:
                    posiciones["mizqb"].append((x, y))
                else:
                    posiciones["mderb"].append((x, y))
            else:
                if x < x_rio:
                    posiciones["mizq"].append((x, y))
                else:
                    posiciones["mder"].append((x, y))

        return posiciones

    def __init__(self):
        self.COLOR_CANIBAL = self.hex_to_bgr(self.HEX_CANIBAL)
        self.COLOR_MISIONERO = self.hex_to_bgr(self.HEX_MISIONERO)
        self.COLOR_BOTE = self.hex_to_bgr(self.HEX_BOTE)
        self.COLOR_RIO = self.hex_to_bgr(self.HEX_RIO)

    def getPercept(self):
        return self.detectar_elementos()

time.sleep(2)

agent = Agent()
env = Environment()

estado = ()

while estado not in estados_terminar:
    percept = env.getPercept()
    estado = agent.compute(percept)