import random
import uuid

import numpy as np
import pandas as pd
import shapely.geometry as geom
from loguru import logger

from .angle import Angle


def decision(probability):
    """
    Функция для получения True переменно с необходимой вероятностью
    :param probability: вероятность с которой необходимо возвращать True
    """
    return random.random() < probability


class Particle:
    def __init__(self, init_coordinates, abs_velocity=0.046, step_counter=0):
        self.trace_data = None
        self._act_state = None
        self._coord = init_coordinates
        self.path = np.vstack((np.empty((0, 3)), self.coord))  # изменение для теста с кубиками
        self._v_abs = abs_velocity
        self.calculation_final = False
        self._cell_id = uuid.uuid4()
        self.phi = Angle(int(str(self.cell_id.int)[::5]), low_border=0, high_border=2 * np.pi)
        self.theta = Angle(int(str(self.cell_id.int)[1::5]), low_border=0, high_border=np.pi)
        self.step_counter = step_counter
        self.max_step_counter = 6

    @property
    def coord(self):
        return self._coord

    def set_coordinate(self, value):
        self._coord = value

    @property
    def v_res(self):
        return self.v_abs * self.v_dir

    @property
    def v_dir(self):
        if self.calculation_final:
            direction = np.array([np.cos(self.phi.value) * np.sin(self.theta.value),
                                  np.sin(self.phi.value) * np.sin(self.theta.value),
                                  np.cos(self.theta.value)])
        else:
            direction = np.array([np.cos(self.phi.prev_value) * np.sin(self.theta.prev_value),
                                  np.sin(self.phi.prev_value) * np.sin(self.theta.prev_value),
                                  np.cos(self.theta.prev_value)])
        return direction

    @property
    def v_abs(self):
        return self._v_abs

    def set_v_abs(self, value):
        self._v_abs = value

    @property
    def cell_id(self):
        return self._cell_id

    def make_step(self, t_step=20):
        new_coordinate = self.coord + self.v_res * t_step
        self.set_coordinate(new_coordinate)
        self.path = np.vstack((self.path, new_coordinate))
        self.step_counter += 1

    def void_test(self):
        self.calculation_final = False
        ind_phi, ind_theta = self.phi.gen_angle(), self.theta.gen_angle()
        while not self.calculation_final:
            if isinstance(ind_phi, int) and isinstance(ind_theta, int):
                self.calculation_final = True
            else:
                ind_phi, ind_theta = self.phi.gen_angle(), self.theta.gen_angle()

    def movement(self):
        if self.step_counter < self.max_step_counter:
            self.make_step()
        else:
            self.void_test()
            if self.calculation_final:
                self.phi.make_angle()
                self.theta.make_angle()
                self.make_step()
            self.step_counter = 0

    def write_path(self, f_path):
        df = pd.DataFrame(self.path, columns=["x", "y", "z"])
        df['act_state'] = self._act_state
        df.to_csv(f_path, sep=';', index=False)

    def write_data(self, f_path):
        df = pd.DataFrame(self.trace_data, columns=["time", "x", "y", "z", "act_state", "act_time", "area", "act_zone"])
        df.to_csv(f_path, sep=';', index=False)


class BasicCell(Particle):
    def __init__(self, basic_cell_init_coordinates,
                 basic_cell_abs_velocity=0.046,
                 basic_cell_step_counter=0,
                 lymph_data_map=None):
        super().__init__(init_coordinates=basic_cell_init_coordinates,
                         abs_velocity=basic_cell_abs_velocity,
                         step_counter=basic_cell_step_counter)
        self.data_map = lymph_data_map or {"red": np.arange(self.phi.values_pull_size // 2)}
        self.bad_angle_set = set()
        self.step_iterations_limit = self.theta.values_pull_size * self.phi.values_pull_size

    def void_test(self):
        self.calculation_final = False
        ind_phi, ind_theta = self.phi.gen_angle(), self.theta.gen_angle()
        while not self.calculation_final:
            if (ind_phi, ind_theta) in self.bad_angle_set:
                ind_phi, ind_theta = self.phi.gen_angle(), self.theta.gen_angle()
            else:
                if (ind_phi in self.data_map["red"]) and (ind_theta in self.data_map["red"]):
                    self.calculation_final = True
                else:
                    self.bad_angle_set.add((ind_phi, ind_theta))
                    ind_phi, ind_theta = self.phi.gen_angle(), self.theta.gen_angle()


class BCell(BasicCell):
    def __init__(self, b_cell_init_coords,
                 b_cell_abs_velocity=0.046,
                 b_cell_step_counter=0,
                 data_map=None,
                 act_state=None,
                 activate=False,
                 act_time=None,
                 curr_act_time=None):
        super(BCell, self).__init__(basic_cell_init_coordinates=b_cell_init_coords,
                                    basic_cell_abs_velocity=b_cell_abs_velocity,
                                    basic_cell_step_counter=b_cell_step_counter,
                                    lymph_data_map=data_map)
        self.cell_type = "B-Cell"
        self._act_state = ["inactive"] or [act_state]  # изменения для теста с кубиками (сейчас всё норм)
        self.activate = activate
        self.orig_phi = self.phi.index
        self.orig_theta = self.theta.index
        self.pre_point = self.coord
        self.act_time = act_time
        self.curr_act_time = curr_act_time or 0
        # по идее случайное число от 4320 до 8640 (это количество 20ти секундных интервалов в 24-48 часах)
        self.act_max_time = 8640
        self.first_act_zone = None
        self.trace_act_zone = list()
        # расстояние до той фолликулы к границе которой идёт В-клетка, рассчитывается (и фолликула и расстояние)
        # в момент попадания клетки на расстояние 140мкм от границы
        self.fol_loop = None

        # теперь это параметр для активированных клеток ,отвечает за нахождение на границе и имеет обратное значение False -- клетка на границе, True -- может отходить от неё
        self.ext_red_loop = None
        # меняет своё значение если клетка после активации в первые три часа попала на расстояние в 5мкм от капсулы,
        # причём она находится в этом коридоре пока не пройдёт 3 часа с момента активации
        self.scs_loop = False
        self.trace_data = list()  # переменная для записи данных траектории в файл
        self.prev_phi_ind = None  # переменная для сохранения предыдущего угла phi
        self.prev_theta_ind = None  # переменная для сохранения предыдущего угла theta

        # ниже цирковой костыль для расчёта вероятности по экспоненте
        # a = 49.9112268932896
        # b = -0.023526445676891473
        # x_1 = np.array([i / 180 for i in range(0, 20000, 20)])
        # y_1 = a * np.exp(b * x_1)

        self.test_flag = False

        # p = 12
        # k = 35
        # b = -0.0383653411210808
        # x_1 = np.array([i / 180 for i in range(0, 20000)])
        # y_1 = p + k * np.exp(b * x_1)
        # self.exp_data = (1 - (y_1 - y_1[-1]) / ((y_1 - y_1[-1])[0]))

        # экспонента для отхода от Т/B границы
        self.exp_data = 0.015/180

        t = np.array([i for i in range(3*60*360)])
        # self.ext_red_exp = 1 - np.exp(-0.06 * t / (3 * 60))

        # экспонента инактивации

        self.inact_exp = 0.04/180  # было хорошо при 0.1 и старте на 8ми часах
        # ниже параметры для клеточной смерти
        self.dead_state = False
        self.act_counter = 0
        t = np.array([i for i in range(3*60*360)])
        # self.dead_exp = 1 - np.exp(-np.log(2)*t/(3*60*12))
        self.dead_exp = np.log(2)*0/(3*60)

        # ниже параметры для столкновений В- и Т-клеток
        self.meet_state = False

    @property
    def act_state(self):
        return self._act_state

    def set_state(self, value):
        self._act_state.append(value)

    def check_lymph(self, scs_data, ext_red_data, t_step: int = 120):
        """
        Генерация точки для нового шага, а также проверка на нахождение этой точки сперва в пределах (z_min, z_max),
        а затем в пределах xy-границы лимфоузла.
        :param scs_data: z-стек xy-границ лимфоузла
        :param ext_red_data: z-стек xy-границ Т/В-зоны лимфоузла
        :param t_step: время прохода по прямой с ново сгенерированными углами, по умолчанию 120 секунд
        :return: индексы подходящих точек в порядке (ind_phi, ind_theta). Возвращает индексы-маркеры
        ошибки (-1, -1), если проверка не допускает перемещения в точку, соответствующую этим углам.
        Пока не доделано, нужно добавить тоннель на границе фолликулы
        """
        # генерируем предварительные углы
        ind_phi, ind_theta = self.phi.gen_angle(), self.theta.gen_angle()
        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                     f"{len(self.path)}: {ind_phi}, {ind_theta}; BAS {len(self.bad_angle_set)}")
        # сразу же проверяем, не были ли они сгенерированы ранее
        if (ind_phi, ind_theta) in self.bad_angle_set:
            # если были, то сразу возвращаем индексы-маркеры ошибки
            logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                         f"{len(self.path)}: {ind_phi}, {ind_theta} in BAS")
            return -1, -1
        # если сгенерированные углы не были отброшены ранее, то запускаем проверки
        else:
            # так как мы сейчас станем проводить предварительные расчеты, то значение финала False
            self.calculation_final = False
            # затем рассчитываем следующую точку на основе этих углов
            point = self.coord + self.v_res * t_step
            # модифицируем координату z для того, чтобы укладываться в шаг картинок
            # point[2] = round(point[2] / 0.4) * 0.4
            # так как на данном этапе рассматривается не весь диапазон картинок, z_min и z_max фиксируются вручную
            z_min = 0.0  # минимальная граница, по умолчанию
            z_max = (len(scs_data) - 1) * 0.4  # максимальная граница, по умолчанию z_min + (len(scs_data) - 1) * 0.4
            # если z-составляющая сгенерированной точки находится в пределах рассматриваемого диапазона лимфоузла,
            if z_min < round(point[2] / 0.4) * 0.4 < z_max:
                # то мы делаем из точки координатной объект geom.Point для проверки нахождения в границах лимфоузла
                p = geom.Point(point[0], point[1])
                # костыль на количество картинок, мы достаём нужный нам срез
                scs_bord = scs_data[int(round(point[2] / 0.4) - z_min / 0.4)]
                # если сгенерированная точка выходит за xy границы лимфоузла,
                if not p.within(scs_bord[0]):
                    # то значит, что точка некорректная
                    logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                 f"{len(self.path)}: {ind_phi}, {ind_theta} in SCS")
                    # записываем индексы углов, приводящие к ошибке в множество
                    self.bad_angle_set.add((ind_phi, ind_theta))
                    # возвращаем индексы-маркеры ошибки
                    return -1, -1
                # если клетка не активирована, то весь подбор заканчивается,
                # если же она активирована то начинается цирк ниже
                elif self.activate:

                    # ряд параметров которые означают различные временные промежутки жизни клетки после активации

                    # время "замирания" сразу после активации, по идее от 30ти до 60ти минут (последний множитель)
                    # в промежутке между 1м и 2м клетки двигаются к scs
                    par_1 = 20 * 3 * 15
                    # после этого момента случайное блуждание пока не окажутся в 140 мкм от границы фолликулы
                    par_2 = 20 * 3 * 180
                    # время к которому они должны оказаться на границе фолликул, то есть в Т/В - зоне
                    par_3 = 20 * 3 * 60 * 6
                    curr_time = (len(self.path) - self.act_time) * 20

                    # первый этап после активации (до par_1 клетка стоит)
                    if par_1 <= curr_time < par_2:
                        # клетка должна после активации мигрировать как наивная, пока не попадёт на
                        # определённое расстояние от scs (не помню точно параметр, но это не так важно)
                        # идёт проверка той точки в которой клетка находится сейчас
                        # "p" - новая точка, "p_2" - точка в которой клетка находится сейчас
                        point_2 = self.path[-1]
                        scs_bord_2 = scs_data[int(round(point_2[2] / 0.4) - z_min / 0.4)]

                        p_2 = geom.Point(point_2[0], point_2[1])
                        logger.debug(f"{self.cell_type}:{self.cell_id} > " +
                                     f"{len(self.path)}: {p_2.distance(scs_bord_2[1])}, {p_2}, {self.scs_loop} ; PAR 1; SEEKING")
                        if self.scs_loop or p_2.distance(scs_bord_2[1]) <= 8:
                            # проверка, если клетка ещё не попала в эту зону,
                            # то при попадании ставится True, если уже тут, то идём дальше по коду
                            self.scs_loop = True
                            # проверка новой точки на нахождение в том самом коридоре
                            if p.distance(scs_bord[1]) <= 8:
                                # проверка на изменение скорости и переизбытка времени активированного состояния
                                self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                                                  self.data_map["med_data"])
                                self.calculation_final = True
                                return ind_phi, ind_theta
                            else:
                                # не подошла точка, пересчитываем
                                logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                             f"{len(self.path)}: {p.distance(scs_bord[1])}, {ind_phi}, {ind_theta}; PAR 1; too far to SCS board")
                                self.bad_angle_set.add((ind_phi, ind_theta))
                                return -1, -1
                        else:
                            # проверка на изменение скорости и переизбытка времени активированного состояния
                            self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                                              self.data_map["med_data"])
                            self.calculation_final = True
                            return ind_phi, ind_theta

                            # это работало, сейчас меняем
                        # if p_2.distance(scs_bord[1]) <= 8 or self.dist_to_scs: # "1" этот параметр будет меняться
                        #     self.dist_to_scs = p_2.distance(scs_bord[1])
                        #     if p.distance(scs_bord[1]) <= 8:
                        #         self.calculation_final = True
                        #         return ind_phi, ind_theta
                        #     else:
                        #         return -1, -1
                        # else:
                        #     self.calculation_final = True
                        #     return ind_phi, ind_theta

                    # второй этап активации (сначала наивное движение,
                    # пока не попадёт в зону в 140 мкм от фолликулы)
                    elif par_2 <= curr_time < par_3:
                        # убираем параметр притяжения к scs
                        self.scs_loop = False
                        # red_bord = red_data[int(round(point[2] / 0.4) - z_min / 0.4)]
                        point = self.coord + self.v_res * t_step
                        point_2 = self.path[-1]
                        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                     f"{len(self.path)}: {self.coord}, {self.path[-1]}, {self.coord == self.path[-1]}; PAR 2; find fckn mistake_v_0")

                        p_curr = geom.Point(point[0], point[1])
                        p_curr_2 = geom.Point(point_2[0], point_2[1])

                        ext_red_bord = ext_red_data[int(round(point[2] / 0.4) - z_min / 0.4)]
                        ext_red_bord_2 = ext_red_data[int(round(point_2[2] / 0.4) - z_min / 0.4)]
                        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                     f"{len(self.path)}: {p_curr.distance(ext_red_bord[0])}, {p_curr_2.distance(ext_red_bord_2[0])}, {point}, {point_2}; PAR 2; find fckn mistake")

                        if p_curr.distance(ext_red_bord[0]) <= 140 * 0.4:
                            if not self.fol_loop:
                                self.fol_loop = True
                            if decision(0.6):
                                if p_curr.distance(ext_red_bord[0]) <= p_curr_2.distance(ext_red_bord_2[0]) or p_curr.distance(ext_red_bord[0]) <= 1:
                                    self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                                                      self.data_map["med_data"])
                                    self.calculation_final = True
                                    logger.debug(f"{self.cell_type}:{self.cell_id} > " +
                                                 f"{len(self.path)}: {p_curr.distance(ext_red_bord[0])}, {p_curr_2.distance(ext_red_bord_2[0])}; PAR 2; good_distance")
                                    return ind_phi, ind_theta
                                else:
                                    logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                                 f"{len(self.path)}: {p_curr.distance(ext_red_bord[0])}, {p_curr_2.distance(ext_red_bord_2[0])}, {point}; PAR 2; bad distance_part_1")

                                    self.bad_angle_set.add((ind_phi, ind_theta))
                                    return -1, -1
                            else:
                                self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                                                  self.data_map["med_data"])
                                self.calculation_final = True
                                logger.debug(f"{self.cell_type}:{self.cell_id} > " +
                                             f"{len(self.path)}: {p_curr.distance(ext_red_bord[0])}, {p_curr_2.distance(ext_red_bord_2[0])}; PAR 2; good_distance")
                                return ind_phi, ind_theta

                        elif self.fol_loop:
                            logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                         f"{len(self.path)}: {p_curr.distance(ext_red_bord[0])}, {p_curr_2.distance(ext_red_bord_2[0])}, {point}; PAR 2; bad distance_part_2")

                            self.bad_angle_set.add((ind_phi, ind_theta))
                            return -1, -1
                        else:

                            self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                                              self.data_map["med_data"])
                            self.calculation_final = True
                            logger.debug(f"{self.cell_type}:{self.cell_id} > " +
                                         f"{len(self.path)}: {p_curr.distance(ext_red_bord[0])}; PAR 2; not_in_fol_loop_good_distance")
                            return ind_phi, ind_theta

                        # расчет для получения индекса фолликулы к которой идти и расстояния
                        # до которой проверять на приближение к 140 мкм
                        # if not self.dist_to_fol:  # расстояние до границы своей фолликулы
                        #     # "p" - новая точка, "p_2" - точка в которой клетка находится сейчас
                        #     point_2 = self.path[-1]
                        #     p_2 = geom.Point(point_2[0], point_2[1])
                        #     red_bord_2 = red_data[int(round(point_2[2] / 0.4) - z_min / 0.4)]
                        #     # словарь с индексами фолликул на срезе и расстояния до
                        #     # каждой из них чтобы найти минимум и проверять приближение  к нему
                        #     pool_dist = dict()
                        #     # создание словаря ключ -- расстояние, значение -- индекс фолликулы
                        #     for bord in red_bord_2.keys():
                        #         pool_dist[p_2.distance(red_bord_2[bord][1][0])] = bord
                        #     # проверяем нашлись ли фолликулы на срезе
                        #     logger.debug(f"{self.cell_type}:{self.cell_id} > " +
                        #                  f"{len(self.path)}: {pool_dist}, {point_2}; PAR 2; init fol_mem")
                        #     if len(pool_dist.keys()) > 0:
                        #         # если нашлись, то устанавливаем self.dist_to_fol
                        #         # [расстояние до ближайшей фолликулы, индекс этой фолликулы]
                        #         self.dist_to_fol = [min(pool_dist.keys()), pool_dist[min(pool_dist.keys())]]
                        #     else:
                        #         # если не нашлись, то забиваем, найдём на следующей попытке
                        #         self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                        #                           self.data_map["med_data"])
                        #         self.calculation_final = True
                        #         return ind_phi, ind_theta
                        # logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                        #              f"{len(self.path)}: {self.dist_to_fol}, {point}; PAR 2; bad distance")
                        # # проверка расстояния
                        # if self.dist_to_fol[0] < 140 * 0.4:
                        #     # если клетка на нужном расстоянии то проверяем,
                        #     # чтобы не уходила (очень похоже что ошибка здесь)
                        #     point_2 = self.path[-1]
                        #     p_2 = geom.Point(point_2[0], point_2[1])
                        #     ext_red_bord_2 = ext_red_data[int(round(point_2[2] / 0.4) - z_min / 0.4)]
                        #     if self.dist_to_fol[1] not in red_bord.keys() or (p.distance(red_bord[self.dist_to_fol[1]][1][0]) > self.dist_to_fol[0] * 0.75 and p.distance(red_bord[self.dist_to_fol[1]][1][0]) > 5):
                        #
                        #         if (self.dist_to_fol[1] in red_bord.keys()) and p.distance(ext_red_bord[0]) > 0.8 * p_2.distance(ext_red_bord_2[0]):
                        #             logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                        #                          f"{len(self.path)}: {p.distance(red_bord[self.dist_to_fol[1]][1][0])},"
                        #                          f" {self.dist_to_fol[0]}, {p.distance(ext_red_bord[0])},"
                        #                          f" {p_2.distance(ext_red_bord_2[0])}; PAR 2; bad distance_part_1")
                        #             self.bad_angle_set.add((ind_phi, ind_theta))
                        #             return -1, -1
                        #         elif p.distance(ext_red_bord[0]) <= p_2.distance(ext_red_bord_2[0]) * 0.3:
                        #             self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                        #                               self.data_map["med_data"])
                        #             self.calculation_final = True
                        #             logger.debug(f"{self.cell_type}:{self.cell_id} > " +
                        #                          f"{len(self.path)}: {p.distance(ext_red_bord[0])}, {p_2.distance(ext_red_bord_2[0])}; PAR 2; good_distance")
                        #             return ind_phi, ind_theta
                        #         else:
                        #             logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                        #                          f"{len(self.path)}: {self.dist_to_fol}, {point}; PAR 2; bad distance")
                        #
                        #             self.bad_angle_set.add((ind_phi, ind_theta))
                        #             return -1, -1
                        #     elif p.distance(ext_red_bord[0]) <= p_2.distance(ext_red_bord_2[0]) * 0.3:
                        #         self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                        #                           self.data_map["med_data"])
                        #         self.calculation_final = True
                        #         logger.debug(f"{self.cell_type}:{self.cell_id} > " +
                        #                      f"{len(self.path)}: {p.distance(ext_red_bord[0])}, {p_2.distance(ext_red_bord_2[0])}; PAR 2; good_distance_2")
                        #         return ind_phi, ind_theta
                        #     else:
                        #         logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                        #                      f"{len(self.path)}: {self.dist_to_fol}, {point}; PAR 2; bad distance_part_3")
                        #
                        #         self.bad_angle_set.add((ind_phi, ind_theta))
                        #         return -1, -1
                        # else:
                        #     self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                        #                       self.data_map["med_data"])
                        #     self.calculation_final = True
                        #     logger.debug(f"{self.cell_type}:{self.cell_id} > " +
                        #                  f"{len(self.path)}: {self.dist_to_fol[0]}; PAR 2; not_in_fol_loop_good_distance")
                        #     return ind_phi, ind_theta

                    # проверка финального этапа активации
                    elif par_3 <= curr_time and not self.ext_red_loop:
                        # тут кетка должны находится только в ext_red
                        # проходит соответствующая проверка что клетка пыталась идти до фолликулы и в ext_red
                        # red_bord = red_data[int(round(point[2] / 0.4) - z_min / 0.4)]
                        ext_red_bord = ext_red_data[int(round(point[2] / 0.4) - z_min / 0.4)]
                        point_2 = self.path[-1]
                        p_2 = geom.Point(point_2[0], point_2[1])
                        ext_red_bord_2 = ext_red_data[int(round(point_2[2] / 0.4) - z_min / 0.4)]
                        # g = p.distance(ext_red_bord[0]) < p_2.distance(ext_red_bord_2[0])
                        if p.within(ext_red_bord[0]) or p.distance(ext_red_bord[0]) < 7.5:
                            logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                         f"{len(self.path)}:, {p.within(ext_red_bord[0])}, { p.distance(ext_red_bord[0])}, PAR 3")
                            self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                                              self.data_map["med_data"])
                            self.calculation_final = True
                            return ind_phi, ind_theta
                        elif p_2.distance(ext_red_bord_2[0]) >= 7.5:
                            if p.distance(ext_red_bord[0]) < p_2.distance(ext_red_bord_2[0]):
                                logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                             f"{len(self.path)}:, {p.within(ext_red_bord[0])}, {p.distance(ext_red_bord[0])}, PAR 3_par_1")
                                self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                                                  self.data_map["med_data"])
                                self.calculation_final = True
                                return ind_phi, ind_theta
                            else:
                                logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                             f"{len(self.path)}:, {p.within(ext_red_bord[0])}, {p.distance(ext_red_bord[0])}, PAR 3_part_2")
                                self.bad_angle_set.add((ind_phi, ind_theta))
                                return -1, -1
                        else:
                            logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                         f"{len(self.path)}:, {p.within(ext_red_bord[0])}, {p.distance(ext_red_bord[0])}, PAR 3_part_3")
                            self.bad_angle_set.add((ind_phi, ind_theta))
                            return -1, -1
                    else:
                        self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                                          self.data_map["med_data"])
                        self.calculation_final = True
                        return ind_phi, ind_theta
                else:
                    self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                                      self.data_map["med_data"])
                    self.calculation_final = True
                    return ind_phi, ind_theta
            # если z-составляющая сгенерированной точки выходит за пределы диапазона,
            else:
                # то значит, что точка некорректная
                logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                             f"{len(self.path)}: {ind_phi}, {ind_theta}; out of lymph")
                # записываем индексы углов, приводящие к ошибке в множество
                self.bad_angle_set.add((ind_phi, ind_theta))
                # возвращаем индексы-маркеры ошибки
                return -1, -1

    def check_area(self, blue_data, med_data, ind_phi, ind_theta, t_step=120):
        """
        Проверка ТОЛЬКО для В-клеток на нахождение НЕ в Т-зоне и НЕ в медуллярной зоне.
        Не генерирует новых углов, так как всегда запускается после check_lymph.
        :param blue_data: z-стек xy-границ Т-зоны
        :param med_data: z-стек xy-границ медуллярной зоны
        :param ind_phi: индексы углов, которые были сгенерированы на этапе check_lymph
        :param ind_theta: индексы углов, которые были сгенерированы на этапе check_lymph. Нам нужны оба этих индекса,
        чтобы можно было возвращать их в случае успешного прохождения проверок
        :param t_step: время прохода по прямой с ново сгенерированными углами, по умолчанию 120 секунд
        :return: индексы подходящих точек в порядке (ind_phi, ind_theta). Возвращает индексы-маркеры
        ошибки (-1, -1), если проверка не допускает перемещения в точку, соответствующую этим углам.
        """
        # ещё на этапе проверки нахождения в лимфоузле были сгенерированы предварительные значения,
        # так как при успешном прохождении проверки нахождения в лимфоузле значения финала изменяются на True,
        # то перед новой проверкой меняем их на False
        self.calculation_final = False
        point = self.coord + self.v_res * t_step
        # модифицируем координату z для того, чтобы укладываться в шаг картинок
        point[2] = round(point[2] / 0.4) * 0.4
        # получаем объект geom.Point для 2D проверок в xy
        p = geom.Point(point[0], point[1])
        # костыльно настраиваем z_min
        z_min = 0.0  # минимальная граница, по умолчанию 0
        # а также достаем xy-границы медуллярной и Т-зон из стека границ
        blue_bord = blue_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        med_bord = med_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        # проверяем, не попали ли в Т-зону
        if not p.within(blue_bord[0]) and not p.within(blue_bord[1]):
            # если нет (мы НЕ в Т-зоне), то проверяем, не попали ли в медуллярную зону

            # ТУТ РАНЬШЕ БЫЛА ПРОСТО ПРОВЕРКА НАХОЖДЕНИЯ В МЕДУЛЯРКЕ, ВЕРНИ ПОТОМ ОБРАТНО    vernul

            if p.within(med_bord[0]) or p.within(med_bord[1]):
                # если попали в медуллярную зону, то значит, что точка некорректная
                logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                             f"{len(self.path)}: {ind_phi}, {ind_theta}; bad med bord distance")
                # записываем индексы углов, приводящие к ошибке в множество
                self.bad_angle_set.add((ind_phi, ind_theta))
                # возвращаем индексы-маркеры ошибки
                return -1, -1
            else:
                # если мы НЕ в Т-зоне и НЕ в медуллярной, то успех, завершаем предварительные вычисления
                # а иначе мы передаём значение True для конца подбора точки
                self.calculation_final = True
                # и возвращаем нормальные индексы подходящих углов
                return ind_phi, ind_theta
        else:
            # если попали в Т-зону, то значит, что точка некорректная
            logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                         f"{len(self.path)}: {ind_phi}, {ind_theta}; in blue zone")
            # записываем индексы углов, приводящие к ошибке в множество
            self.bad_angle_set.add((ind_phi, ind_theta))
            # возвращаем индексы-маркеры ошибки
            return -1, -1

    def rotation(self, alpha):
        """
        Функция для поворота вектора скорости на угол
        :param alpha: угол поворота
        """
        ph_st = self.phi.index
        th_st = self.theta.index

        phi = np.array([i for i in range(ph_st - alpha, ph_st + alpha + 1)])
        theta = np.array([i for i in range(th_st - alpha, th_st + alpha + 1)])
        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                     f"{len(self.path)}: {phi}, {theta}, {self.phi.identifier};rotation_1")
        oo = np.where(((alpha * 0.8) ** 2 < (phi - ph_st) ** 2 + (theta - th_st) ** 2) & ((phi - ph_st) ** 2 + (theta - th_st) ** 2 < (alpha * 1.2) ** 2))
        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                     f"{len(self.path)}: {(0,len(oo[0]) - 1)}, {oo}, {self.phi.identifier};rotation_2")
        num_phi = self.phi.gen_angle(pool=len(oo[0]) - 1)
        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                     f"{len(self.path)}: {(0, len(oo[0]) - 1)}, {phi[oo[0][num_phi]]}, {theta[oo[0][num_phi]]}, {self.phi.identifier};rotation_3")
        return phi[oo[0][num_phi]], theta[oo[0][num_phi]]

    def make_pre_step(self, t_step=20):
        self.pre_point = self.coord + self.v_res * t_step

    def pre_step(self):
        """
        Функция "пре-шага" по сути такой же как и обычный,
         но реализована для столкновений и в ней же реализован возврат
        """
        if self.meet_state or self.dead_state:
            self.make_pre_step()
        elif self.step_counter != self.max_step_counter and self.step_counter != 0:
            self.make_pre_step()
        else:
            ind_phi, ind_theta = -1, -1
            tot_attempts_counter = 0  # счётчик для проверки соответствия с self.step_iterations_limit
            # до тех пор, пока проверки возвращают индексы-маркеры ошибки
            if not self.dead_state:
                while (ind_phi, ind_theta) == (-1, -1):
                    # проверяем, не ушли ли мы за допустимое число переборов
                    if len(self.bad_angle_set) < self.step_iterations_limit / 2:
                        # запускаем проверку нахождения в лимфоузле
                        ind_phi, ind_theta = self.check_lymph(self.data_map["scs_data"], self.data_map["ext_red_data"])
                        # если точка находится в лимфоузле, то
                        if (ind_phi, ind_theta) != (-1, -1):
                            # проверяем, находится ли она в пределах допустимых областей
                            ind_phi, ind_theta = self.check_area(self.data_map["blue_data"], self.data_map["med_data"],
                                                                 ind_phi, ind_theta)

                        # при любом исходе увеличиваем счётчик переборов
                        tot_attempts_counter += 1
                    # а вот если число переборов выше допустимого, то сообщаем о ошибке
                    else:
                        # в случае, когда угол не найден, присваиваем прежнее значение угла
                        # (по идее надо обратное, но я дебил upd.: проверил, не помогло, вернул как было)
                        # self.phi.set_pre_index(180 - self.prev_phi_ind)
                        # self.theta.set_pre_index(180 - self.prev_theta_ind)
                        # print(self.trace_data[-1])
                        # if (self.phi.pre_index is None) or (self.theta.pre_index is None):
                        #     print(f"No way found from the start {self.coord}")
                        #     exit()
                        # (ind_phi, ind_theta) = (0, -1)
                        raise ValueError("Area check error occurred " +
                                         "for cell {0} at {1}".format(self.cell_id, self.coord))
                    # При переходе на новую итерацию возможны три исхода:
                    # 1) результаты проверок вернули индексы-маркеры ошибки, цикл идёт на новый круг
                    # 2) результаты проверок вернули правильный индекс, можно завершать цикл
                    # 3) новая итерация завершится до проверок, так как был превышен допустимый предел итераций,
                    # будет запущена ValueError
                # Перед переходом к непосредственному утверждению угла ещё раз проверим, не вышло ли так, что
            # прошедший проверки угол был каким-либо образом ранее определен в ошибочные
            if (ind_phi, ind_theta) in self.bad_angle_set:
                self.write_path('error' + "{}.csv".format(self.cell_id))
                self.write_data('error' + "data_{}.csv".format(self.cell_id))
                raise ValueError("Area check error occurred " +
                                 "for cell {0} at {1}".format(self.cell_id, self.coord))
            # когда найдена удовлетворяющая точка, calculation_final точно True,
            # в таком случае можем двигаться по прямой
            logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                         f"{len(self.path)}: {ind_phi}, {ind_theta} ; calc_final; per_cell")
            if self.calculation_final:
                self.phi.make_angle()
                self.theta.make_angle()

                self.prev_phi_ind = self.phi.pre_index
                self.prev_theta_ind = self.theta.pre_index
                self.make_pre_step()
                self.step_counter = 0
                # если мы решим что поворачиваем единожды за попытку поворота, то добавить изменение step_counter

    def movement(self):
        if self.activate:
            self.act_changes()

        self.bad_angle_set = set()
        new_coordinate = self.pre_point
        self.set_coordinate(new_coordinate)
        self.path = np.vstack((self.path, new_coordinate))
        self.step_counter += 1

        self.orig_phi = self.phi.index
        self.orig_theta = self.theta.index

        if self.step_counter == self.max_step_counter:
            self.step_counter = 0

        # self.check_active(self.data_map["scs_data"], self.data_map["IF_data"], self.data_map["med_data"])

        # store data
        area = self.curr_area(self.data_map["scs_data"], self.data_map["IF_data"], self.data_map["med_data"],
                              self.data_map["red_data"], self.data_map["blue_data"], self.data_map["ext_red_data"])
        self.trace_act_zone.append(self.first_act_zone)
        if not self.dead_state and not self.meet_state:
            if self.activate:
                self.trace_data.append(
                    [len(self.path) * 20, self.path[-1][0], self.path[-1][1], self.path[-1][2], self.activate,
                     (len(self.path) - self.act_time) * 20, area, self.first_act_zone])
                self.set_state("active")
            else:
                self.trace_data.append(
                    [len(self.path) * 20, self.path[-1][0], self.path[-1][1], self.path[-1][2], self.activate, 0, area,  self.first_act_zone])
                self.set_state("0")
        elif self.dead_state:
            self.trace_data.append(
                [len(self.path) * 20, self.path[-1][0], self.path[-1][1], self.path[-1][2], self.activate, 0, area,  self.first_act_zone])
            self.set_state("dead")
        else:
            self.trace_data.append(
                [len(self.path) * 20, self.path[-1][0], self.path[-1][1], self.path[-1][2], self.activate, 0, area,  self.first_act_zone])
            self.set_state("meet")

    def check_active(self, point, scs_data, if_data, med_data):
        """
        Функция проверки активации
        :param point: точка для проверки
        :param scs_data: z-стек xy-границ лимфоузла
        :param if_data: z-стек xy-границ интер фолликулярных зон лимфоузла
        :param med_data: z-стек xy-границ медуллярной зоны лимфоузла
        """

        pass

        # z_min = 0
        # # print(point, int(round(point[2] / 0.4) - z_min / 0.4))
        # scs_bord = scs_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        # if_zone = if_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        # med_zone = med_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        #
        # p = geom.Point(point[0], point[1])
        #
        # par_2 = 20 * 3 * 60 * 3
        # par_4 = 20 * 3 * 60 * 12
        #
        # # разные сценарии для активированной и нет клетки
        # if self.activate and not self.test_flag:
        #     # изменения, связанные с активацией (скорость и тд)
        #
        #     if not self.act_time:
        #         self.act_time = len(self.path)
        #     curr_time = (len(self.path) - self.act_time) * 20
        #
        #     if self.act_time:
        #         curr_time = (len(self.path) - self.act_time) * 20
        #         if curr_time >= par_4:
        #             if p.distance(if_zone[0]) == 5:
        #                 self.curr_act_time = len(self.path)
        #                 if par_2 <= curr_time:
        #                     self.act_counter += 1
        #
        #     # проверка не находится ли клетка активированной слишком долго
        #     # если попала в IF, то активация
        #     if p.within(if_zone[0]) or p.within(if_zone[1]):
        #         self.activate = True
        #
        #         self.curr_act_time = len(self.path)
        #         if par_2 <= curr_time:
        #             self.act_counter += 1
        #
        #     # если близко к медуллярной зоне, то активация
        #     elif p.distance(med_zone[0]) < 2:
        #         self.activate = True
        #         # self.set_state("active")
        #         self.curr_act_time = len(self.path)
        #         if par_2 <= curr_time:
        #             self.act_counter += 1
        #         #  если близко к капсуле, то активация
        #     elif p.distance(scs_bord[1]) < 2:
        #         self.activate = True
        #         # self.set_state("active")
        #         self.curr_act_time = len(self.path)
        #         if par_2 <= curr_time:
        #             self.act_counter += 1
        #
        #     # if self.activate and len(self.path) - self.act_time >= self.act_max_time:
        #     #     self.activate = False
        #     #     self.act_counter = 0
        #     #     self.curr_act_time = None
        #     #     self.act_time = None
        #
        # elif not self.meet_state and not self.dead_state and not self.test_flag:
        #     # если не активирована, проверяем не попала ли куда
        #     # self.curr_act_time = len(self.path)
        #     # похоже на костыль но нужный
        #     # if not self.act_time:
        #     #     self.act_time = 0
        #     # если попала в IF, то активация
        #     if p.within(if_zone[0]) or p.within(if_zone[1]):
        #         self.activate = True
        #
        #         self.curr_act_time = len(self.path)
        #         self.act_counter += 1
        #         if not self.act_time:
        #             self.act_time = len(self.path)
        #         if not self.first_act_zone:
        #             self.first_act_zone = 'IF'
        #         logger.debug(f"{self.cell_type}:{self.cell_id} > " +
        #                      f"{len(self.path)}: {self.act_time}, {self.first_act_zone} ;cell activate")
        #     # если близко к медуллярной зоне, то активация
        #     elif p.distance(med_zone[0]) < 2:
        #         self.activate = True
        #         # self.set_state("active")
        #         self.curr_act_time = len(self.path)
        #         self.act_counter += 1
        #         if not self.act_time:
        #             self.act_time = len(self.path)
        #             #  если близко к капсуле, то активация
        #         if not self.first_act_zone:
        #             self.first_act_zone = 'med'
        #         logger.debug(f"{self.cell_type}:{self.cell_id} > " +
        #                      f"{len(self.path)}: {self.act_time}, {self.first_act_zone} ;cell activate")
        #     elif p.distance(scs_bord[1]) < 2:
        #         self.activate = True
        #         # self.set_state("active")
        #         self.curr_act_time = len(self.path)
        #         self.act_counter += 1
        #         if not self.act_time:
        #             self.act_time = len(self.path)
        #
        #         if not self.first_act_zone:
        #             self.first_act_zone = 'scs'
        #         logger.debug(f"{self.cell_type}:{self.cell_id} > " +
        #                      f"{len(self.path)}: {self.act_time}, {self.first_act_zone} ;cell activate")
        #     # else:
        #     #     self.set_state("inactive")
        #
        # # теперь нужна функция модуляции скорости и угла в зависимости от таймера и расположения внутри фолликулы

    def act_changes(self):
        """
        Функция изменения скорости при разных этапах активации
        """
        pass
        # # ряд параметров которые означают различные временные промежутки жизни клетки после активации
        # # время "замирания" сразу после активации, по идее от 2x до 30ти минут (последний множитель)
        # # в промежутке между 1м и 2м клетки двигаются к scs
        # par_1 = 20 * 3 * 15
        # # после этого момента случайное блуждание пока не окажутся в 140 мкм от границы фолликулы
        # par_2 = 20 * 3 * 180
        # # время к которому они должны оказаться на границе фолликул, то есть в Т/В - зоне
        # par_3 = 20 * 3 * 60 * 6
        #
        # par_4 = 20 * 3 * 60 * 12
        #
        # par_4_5 = 20 * 3 * 60 * 12
        #
        # par_5 = 20 * 3 * 60 * 24
        # curr_time = (len(self.path) - self.act_time) * 20
        #
        # if curr_time < par_1:
        #     self.set_v_abs(0)
        #     # print('0')
        # # v_abs = 0
        # elif par_1 <= curr_time < par_2:
        #     self.set_v_abs(0.046 * 0.5)
        #     # print('1')
        # elif par_2 <= curr_time < par_3:
        #     self.set_v_abs(0.046)
        #     # print('2', end=',')
        #
        # # elif par_4 <= curr_time:
        # #     self.set_v_abs(0.046 * 1.5)
        #     # print('\n see:', self.act_time, self.curr_act_time,
        #     #       decision(self.exp_data[int((len(self.path) - self.curr_act_time) / 20)]))
        #
        #     # В первом и третьем IF self.curr_act_time изменён на self.act_time для тестов
        # elif par_3 <= curr_time:
        #
        #     if not self.dead_state:
        #         logger.trace(f"{self.cell_type}:{self.cell_id} > " +
        #                      f"{len(self.path)}: {self.act_time}, {int((len(self.path) - self.act_time))};cell check")
        #         if decision(self.exp_data) and not self.ext_red_loop:
        #             self.ext_red_loop = True
        #             logger.debug(f"{self.cell_type}:{self.cell_id} > " +
        #                          f"{len(self.path)}: {self.act_time}, {int((len(self.path) - self.act_time))}, {self.exp_data} ;cell unloop")
        #             if par_4_5 <= curr_time and decision(self.inact_exp):
        #                 logger.debug(f"{self.cell_type}:{self.cell_id} > " +
        #                              f"{len(self.path)}: {self.act_time}, {int((len(self.path) - self.act_time))}, {self.inact_exp} ;cell inactivate")
        #                 self.activate = False
        #                 self.act_time = None
        #                 self.curr_act_time = None
        #                 self.fol_loop = False
        #                 self.ext_red_loop = False
        #                 self.act_counter = 0
        #                 self.set_v_abs(0.046)
        #                 self.test_flag = True
        #         elif par_4_5 <= curr_time and decision(self.inact_exp):
        #             logger.debug(f"{self.cell_type}:{self.cell_id} > " +
        #                          f"{len(self.path)}: {self.act_time}, {int((len(self.path) - self.act_time))}, {self.inact_exp} ;cell inactivate")
        #             self.activate = False
        #             self.act_time = None
        #             self.curr_act_time = None
        #             self.fol_loop = False
        #             self.ext_red_loop = False
        #             self.act_counter = 0
        #             self.set_v_abs(0.046)
        #             self.test_flag = True
        #
        #         if par_4 <= curr_time:
        #             self.set_v_abs(0.046 * 1.5)
        #
        #     if par_5 <= curr_time and decision(self.dead_exp) and self.act_counter > 1:
        #         logger.debug(f"{self.cell_type}:{self.cell_id} > " +
        #                      f"{len(self.path)}: {self.act_time} ;cell RIP")
        #         self.dead_state = True
        #         self.act_counter = 0
        #         self.set_v_abs(0.046 * 0)
        #         self.activate = False
        #         self.ext_red_loop = False
        #         self.act_time = None
        #         self.curr_act_time = None
        #         self.test_flag = True
        #
        # # print('3')
        #
        # # if self.activate and len(self.path) - self.act_time >= self.act_max_time:
        # #     self.activate = False
        # #     self.curr_act_time = None
        # #     self.act_time = None

    # v_abs = стандартное значение, плюс модуляция угла в сторону сцс
    # появляется модуляция угла в сторону Т/В границы (по сути границы фолликулы)

    # Вопросы которые надо обсудить
    # 1. как модулируется угол, то что написано не подходит (либо я тупой)
    # так как сейчас это реализовано как урезание угла на какую-то величину (не понятно как её рассчитывать,
    # и если честно идей ноль, можно придумать что-тио с расчётом расстояния
    # до нужной границы, но что делать если клетка подошла вплотную,
    # я не могу родить математический расчёт для этого)

    # 2. норм ли будет каждый раз проверять все фолликулы на этом срезе и
    # что делать с вертикальной границей? (или лучше сделать отдельный параметр и проверять его в других модулях)

    # 3. как правильно поменять значение угла (плюс перепроверь как меняется это
    # значение, насколько я помню они не мгновенно ускоряются)

    # 4. нам нужно сделать "туннель" на границе и я не ебу как

    #  P.S. в моём понимании эта функция используется в конце шага и
    #  проверяет эти параметры накладывая ограничения и тд
    def curr_area(self, scs_data, if_data, med_data, red_data, blue_data, ext_red_data):
        """
        Функция для записи зоны нахождения в данные
        """

        if self.dead_state:
            return 'dead'
        elif self.meet_state:
            return 'meet'

        point = self.path[-1]
        z_min = 0
        p = geom.Point(point[0], point[1])
        scs_zone = scs_data[int(round(point[2] / 0.4) - z_min / 0.4)][1]
        if_zone = if_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        med_zone = med_data[int(round(point[2] / 0.4) - z_min / 0.4)][0]
        red_zone = red_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        ext_red_zone = ext_red_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        _c = ''

        if p.within(if_zone[0]) or p.within(if_zone[1]):
            # for fol in red_zone.keys():
            #     if p.within(red_zone[fol][0][0]):
            #         # print(p.distance(red_zone[fol][1][0]))
            #         if 140 * 0.4 < p.distance(red_zone[fol][1][0]) < 200 * 0.4:
            #             _c = '_140_200_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]
            #         elif 20 * 0.4 < p.distance(red_zone[fol][1][0]) < 80 * 0.4:
            #             _c = '_20_80_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]

            return 'IF'
        elif p.distance(scs_zone) < 5:
            # for fol in red_zone.keys():
            #     if p.within(red_zone[fol][0][0]):
            #         # print(p.distance(red_zone[fol][1][0]))
            #         if 140 * 0.4 < p.distance(red_zone[fol][1][0]) < 200 * 0.4:
            #             _c = '_140_200_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]
            #         elif 20 * 0.4 < p.distance(red_zone[fol][1][0]) < 80 * 0.4:
            #             _c = '_20_80_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]
            return 'scs'
        elif p.distance(med_zone) < 2:
            return 'med'
        elif p.within(ext_red_zone[0]) or p.within(ext_red_zone[1]):
            for fol in red_zone.keys():
                if p.within(red_zone[fol][0][0]):
                    # print(p.distance(red_zone[fol][1][0]))
                    if 140 * 0.4 < p.distance(red_zone[fol][1][0]) < 200 * 0.4:
                        _c = '_140_200_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]
                    elif 20 * 0.4 < p.distance(red_zone[fol][1][0]) < 80 * 0.4:
                        _c = '_20_80_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]

                    return fol[:3]

            return 'ext_red'
        else:
            for fol in red_zone.keys():
                if p.within(red_zone[fol][0][0]):
                    # print(p.distance(red_zone[fol][1][0]))
                    if 140 * 0.4 < p.distance(red_zone[fol][1][0]) < 200 * 0.4:
                        _c = '_140_200_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]
                    elif 20 * 0.4 < p.distance(red_zone[fol][1][0]) < 80 * 0.4:
                        _c = '_20_80_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]

                    return fol[:3]

            return 'ext_red'

    def meet_changes(self):
        # нужно понять надо ли выключать параметры связанные с активацией или можно только activate
        # Upd.: вроде норм поправил в act_changes
        if self.activate and (len(self.path) - self.act_time) * 20 > 20 * 3 * 60 * 6:
            self.meet_state = True
            self.set_v_abs(0.046 * 0)
            self.activate = False
            return True
        else:
            return False


class TCell(BasicCell):
    def __init__(self, t_cell_init_coords,
                 t_cell_abs_velocity=0, # поменял скорость на ноль из-за тестов с В-клетками
                 t_cell_step_counter=0,
                 data_map=None,
                 act_state=None,
                 activate=True,
                 act_time=None,
                 curr_act_time=None):
        super(TCell, self).__init__(basic_cell_init_coordinates=t_cell_init_coords,
                                    basic_cell_abs_velocity=t_cell_abs_velocity,
                                    basic_cell_step_counter=t_cell_step_counter,
                                    lymph_data_map=data_map)
        self.cell_type = "T-Cell"
        self._act_state = ["inactive"] or [act_state]  # изменения для теста с кубиками (сейчас всё норм)
        self.max_step_counter = 8
        self.activate = activate
        self.orig_phi = self.phi.index
        self.orig_theta = self.theta.index
        self.pre_point = self.coord
        self.act_time = act_time
        self.curr_act_time = curr_act_time or 0
        # по идее случайное число от 4320 до 8640 (это количество 20ти секундных интервалов в 24-48 часах)
        self.act_max_time = 8640
        # расстояние до той фолликулы к границе которой идёт В-клетка, рассчитывается (и фолликула и расстояние)
        # в момент попадания клетки на расстояние 140мкм от границы
        self.dist_to_fol = None
        # меняет своё значение если клетка после активации в первые три часа попала на расстояние в 5мкм от капсулы,
        # причём она находится в этом коридоре пока не пройдёт 3 часа с момента активации
        self.scs_loop = False
        self.trace_data = list()  # переменная для записи данных траектории в файл
        self.prev_phi_ind = None  # переменная для сохранения предыдущего угла phi
        self.prev_theta_ind = None  # переменная для сохранения предыдущего угла theta

        # ниже цирковой костыль для расчёта вероятности по экспоненте
        a = 49.9112268932896
        b = -0.023526445676891473
        x_1 = np.array([i / 180 for i in range(2160, 20000)])
        y_1 = a * np.exp(b * x_1)
        self.exp_data = (1 - (y_1 - y_1[-1]) / ((y_1 - y_1[-1])[0]))

        # ниже параметры для столкновений В- и Т-клеток
        self.meet_state = False

    @property
    def act_state(self):
        return self._act_state

    def set_state(self, value):
        self._act_state.append(value)

    def check_lymph(self, scs_data, blue_data, red_data, t_step: int = 120):
        """
        ОПИСАНИЕ ДО СИХ ПОР ДЛЯ В_КЛЕТОК!!!!

        Генерация точки для нового шага, а также проверка на нахождение этой точки сперва в пределах (z_min, z_max),
        а затем в пределах xy-границы лимфоузла.
        :param scs_data: z-стек xy-границ лимфоузла
        :param blue_data: z-стек xy-границ Т/В-зоны лимфоузла
        :param red_data: z-стек xy-границ фолликул лимфоузла формат: [..., {'индекс фолликулы':[границы]}, ...]
        :param t_step: время прохода по прямой с ново сгенерированными углами, по умолчанию 120 секунд
        :return: индексы подходящих точек в порядке (ind_phi, ind_theta). Возвращает индексы-маркеры
        ошибки (-1, -1), если проверка не допускает перемещения в точку, соответствующую этим углам.
        Пока не доделано, нужно добавить тоннель на границе фолликулы
        """
        # генерируем предварительные углы
        ind_phi, ind_theta = self.phi.gen_angle(), self.theta.gen_angle()
        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                     f"{len(self.path)}: {ind_phi}, {ind_theta}; BAS {len(self.bad_angle_set)}")
        # сразу же проверяем, не были ли они сгенерированы ранее
        if (ind_phi, ind_theta) in self.bad_angle_set:
            # если были, то сразу возвращаем индексы-маркеры ошибки
            logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                         f"{len(self.path)}: {ind_phi}, {ind_theta} in BAS")
            return -1, -1
        # если сгенерированные углы не были отброшены ранее, то запускаем проверки
        else:
            # так как мы сейчас станем проводить предварительные расчеты, то значение финала False
            self.calculation_final = False
            # затем рассчитываем следующую точку на основе этих углов
            point = self.coord + self.v_res * t_step
            # модифицируем координату z для того, чтобы укладываться в шаг картинок
            point[2] = round(point[2] / 0.4) * 0.4
            # так как на данном этапе рассматривается не весь диапазон картинок, z_min и z_max фиксируются вручную
            z_min = 0.0  # минимальная граница, по умолчанию
            z_max = (len(scs_data) - 1) * 0.4  # максимальная граница, по умолчанию z_min + (len(scs_data) - 1) * 0.4
            # если z-составляющая сгенерированной точки находится в пределах рассматриваемого диапазона лимфоузла,
            if z_min < point[2] < z_max:
                # то мы делаем из точки координатной объект geom.Point для проверки нахождения в границах лимфоузла
                p = geom.Point(point[0], point[1])
                # костыль на количество картинок, мы достаём нужный нам срез
                scs_bord = scs_data[int(round(point[2] / 0.4) - z_min / 0.4)]
                # если сгенерированная точка выходит за xy границы лимфоузла,
                if not p.within(scs_bord[0]):
                    # то значит, что точка некорректная
                    logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                                 f"{len(self.path)}: {ind_phi}, {ind_theta} in SCS")
                    # записываем индексы углов, приводящие к ошибке в множество
                    self.bad_angle_set.add((ind_phi, ind_theta))
                    # возвращаем индексы-маркеры ошибки
                    return -1, -1
                # если клетка не активирована, то весь подбор заканчивается,
                # если же она активирована то начинается цирк ниже

                # self.check_active(point, self.data_map["scs_data"], self.data_map["IF_data"],
                #                   self.data_map["med_data"])
                self.calculation_final = True
                return ind_phi, ind_theta
            # если z-составляющая сгенерированной точки выходит за пределы диапазона,
            else:
                # то значит, что точка некорректная
                logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                             f"{len(self.path)}: {ind_phi}, {ind_theta}; out of lymph")
                # записываем индексы углов, приводящие к ошибке в множество
                self.bad_angle_set.add((ind_phi, ind_theta))
                # возвращаем индексы-маркеры ошибки
                return -1, -1

    def check_area(self, red_data, med_data, ind_phi, ind_theta, t_step=120):
        """
        ОПИСАНИЕ ДО СИХ ПОР ДЛЯ В_КЛЕТОК!!!!

        Проверка ТОЛЬКО для В-клеток на нахождение НЕ в Т-зоне и НЕ в медуллярной зоне.
        Не генерирует новых углов, так как всегда запускается после check_lymph.
        :param red_data: z-стек xy-границ Т-зоны
        :param med_data: z-стек xy-границ медуллярной зоны
        :param ind_phi: индексы углов, которые были сгенерированы на этапе check_lymph
        :param ind_theta: индексы углов, которые были сгенерированы на этапе check_lymph. Нам нужны оба этих индекса,
        чтобы можно было возвращать их в случае успешного прохождения проверок
        :param t_step: время прохода по прямой с ново сгенерированными углами, по умолчанию 120 секунд
        :return: индексы подходящих точек в порядке (ind_phi, ind_theta). Возвращает индексы-маркеры
        ошибки (-1, -1), если проверка не допускает перемещения в точку, соответствующую этим углам.
        """
        # ещё на этапе проверки нахождения в лимфоузле были сгенерированы предварительные значения,
        # так как при успешном прохождении проверки нахождения в лимфоузле значения финала изменяются на True,
        # то перед новой проверкой меняем их на False
        self.calculation_final = False
        point = self.coord + self.v_res * t_step
        # модифицируем координату z для того, чтобы укладываться в шаг картинок
        point[2] = round(point[2] / 0.4) * 0.4
        # получаем объект geom.Point для 2D проверок в xy
        p = geom.Point(point[0], point[1])
        # костыльно настраиваем z_min
        z_min = 0.0  # минимальная граница, по умолчанию 0
        # а также достаем xy-границы медуллярной и Т-зон из стека границ
        red_bord = red_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        med_bord = med_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        # проверяем, не попали ли в Т-зону

        for fol in red_bord.keys():
            if p.within(red_bord[fol][0][0]):

                logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                             f"{len(self.path)}: {ind_phi}, {ind_theta}; bad fol bord distance")
                # записываем индексы углов, приводящие к ошибке в множество
                self.bad_angle_set.add((ind_phi, ind_theta))
                # возвращаем индексы-маркеры ошибки
                return -1, -1

        if p.within(med_bord[0]) or p.within(med_bord[1]):
            # если попали в медуллярную зону, то значит, что точка некорректная
            logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                         f"{len(self.path)}: {ind_phi}, {ind_theta}; bad med bord distance")
            # записываем индексы углов, приводящие к ошибке в множество
            self.bad_angle_set.add((ind_phi, ind_theta))
            # возвращаем индексы-маркеры ошибки
            return -1, -1
        else:
            # если мы НЕ в Т-зоне и НЕ в медуллярной, то успех, завершаем предварительные вычисления
            # а иначе мы передаём значение True для конца подбора точки
            self.calculation_final = True
            # и возвращаем нормальные индексы подходящих углов
            return ind_phi, ind_theta
        # если мы НЕ в Т-зоне и НЕ в медуллярной, то успех, завершаем предварительные вычисления
        # а иначе мы передаём значение True для конца подбора точки

    def rotation(self, alpha):
        """
        Функция для поворота вектора скорости на угол
        :param alpha: угол поворота
        """
        ph_st = self.phi.index
        th_st = self.theta.index

        phi = np.array([i for i in range(ph_st - alpha, ph_st + alpha + 1)])
        theta = np.array([i for i in range(th_st - alpha, th_st + alpha + 1)])
        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                     f"{len(self.path)}: {phi}, {theta}, {self.phi.identifier};rotation_1")
        oo = np.where(((alpha * 0.8) ** 2 < (phi - ph_st) ** 2 + (theta - th_st) ** 2) & ((phi - ph_st) ** 2 + (theta - th_st) ** 2 < (alpha * 1.2) ** 2))
        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                     f"{len(self.path)}: {(0,len(oo[0]) - 1)}, {oo}, {self.phi.identifier};rotation_2")
        num_phi = self.phi.gen_angle(pool=len(oo[0]) - 1)
        logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                     f"{len(self.path)}: {(0, len(oo[0]) - 1)}, {phi[oo[0][num_phi]]}, {theta[oo[0][num_phi]]}, {self.phi.identifier};rotation_3")
        return phi[oo[0][num_phi]], theta[oo[0][num_phi]]

    def make_pre_step(self, t_step=20):
        self.pre_point = self.coord + self.v_res * t_step

    def pre_step(self):
        """
        Функция "пре-шага" по сути такой же как и обычный,
         но реализована для столкновений и в ней же реализован возврат
        """
        if self.meet_state:
            self.make_pre_step()
        elif self.step_counter != self.max_step_counter and self.step_counter != 0:
            self.make_pre_step()
        else:
            ind_phi, ind_theta = -1, -1
            tot_attempts_counter = 0  # счётчик для проверки соответствия с self.step_iterations_limit
            # до тех пор, пока проверки возвращают индексы-маркеры ошибки
            while (ind_phi, ind_theta) == (-1, -1):
                # проверяем, не ушли ли мы за допустимое число переборов
                if len(self.bad_angle_set) < self.step_iterations_limit / 2:
                    # запускаем проверку нахождения в лимфоузле
                    ind_phi, ind_theta = self.check_lymph(self.data_map["scs_data"], self.data_map["ext_red_data"],
                                                          self.data_map["red_data"])
                    # если точка находится в лимфоузле, то
                    if (ind_phi, ind_theta) != (-1, -1):
                        # проверяем, находится ли она в пределах допустимых областей
                        ind_phi, ind_theta = self.check_area(self.data_map["red_data"], self.data_map["med_data"],
                                                             ind_phi, ind_theta)

                    # при любом исходе увеличиваем счётчик переборов
                    tot_attempts_counter += 1
                # а вот если число переборов выше допустимого, то сообщаем о ошибке
                else:
                    # в случае, когда угол не найден, присваиваем прежнее значение угла
                    # (по идее надо обратное, но я дебил upd.: проверил, не помогло, вернул как было)
                    # self.phi.set_pre_index(180 - self.prev_phi_ind)
                    # self.theta.set_pre_index(180 - self.prev_theta_ind)
                    # print(self.trace_data[-1])
                    # if (self.phi.pre_index is None) or (self.theta.pre_index is None):
                    #     print(f"No way found from the start {self.coord}")
                    #     exit()
                    # (ind_phi, ind_theta) = (0, -1)
                    raise ValueError("Area check error occurred " +
                                     "for cell {0} at {1}".format(self.cell_id, self.coord))
                # При переходе на новую итерацию возможны три исхода:
                # 1) результаты проверок вернули индексы-маркеры ошибки, цикл идёт на новый круг
                # 2) результаты проверок вернули правильный индекс, можно завершать цикл
                # 3) новая итерация завершится до проверок, так как был превышен допустимый предел итераций,
                # будет запущена ValueError
            # Перед переходом к непосредственному утверждению угла ещё раз проверим, не вышло ли так, что
            # прошедший проверки угол был каким-либо образом ранее определен в ошибочные
            if (ind_phi, ind_theta) in self.bad_angle_set:
                self.write_path('error' + "{}.csv".format(self.cell_id))
                self.write_data('error' + "data_{}.csv".format(self.cell_id))
                raise ValueError("Area check error occurred " +
                                 "for cell {0} at {1}".format(self.cell_id, self.coord))
            # когда найдена удовлетворяющая точка, calculation_final точно True,
            # в таком случае можем двигаться по прямой
            logger.trace(f"{self.cell_type}:{self.cell_id} > " +
                         f"{len(self.path)}: {ind_phi}, {ind_theta} ; calc_final; per_cell")
            if self.calculation_final:
                self.phi.make_angle()
                self.theta.make_angle()

                self.prev_phi_ind = self.phi.pre_index
                self.prev_theta_ind = self.theta.pre_index
                self.make_pre_step()
                self.step_counter = 0
                # если мы решим что поворачиваем единожды за попытку поворота, то добавить изменение step_counter

    def movement(self):
        self.bad_angle_set = set()
        new_coordinate = self.pre_point
        self.set_coordinate(new_coordinate)
        self.path = np.vstack((self.path, new_coordinate))
        self.step_counter += 1

        self.orig_phi = self.phi.index
        self.orig_theta = self.theta.index

        if self.step_counter == self.max_step_counter:
            self.step_counter = 0

        self.check_active(self.data_map["scs_data"], self.data_map["IF_data"], self.data_map["med_data"])

        # store data
        area = self.curr_area(self.data_map["scs_data"], self.data_map["IF_data"], self.data_map["med_data"],
                              self.data_map["red_data"], self.data_map["blue_data"], self.data_map["ext_red_data"])
        if self.meet_state:
            self.trace_data.append(
                [len(self.path) * 20, self.path[-1][0], self.path[-1][1], self.path[-1][2], self.activate, 0, area, ''])
            self.set_state("meet")
        elif self.activate:
            self.trace_data.append(
                [len(self.path) * 20, self.path[-1][0], self.path[-1][1], self.path[-1][2], self.activate,
                 (len(self.path) - self.act_time) * 20, area, ''])
            self.set_state("active")
        else:
            self.trace_data.append(
                [len(self.path) * 20, self.path[-1][0], self.path[-1][1], self.path[-1][2], self.activate, 0, area, ''])
            self.set_state("0")

    def check_active(self, scs_data, if_data, med_data):
        """
        Функция проверки активации
        :param scs_data: z-стек xy-границ лимфоузла
        :param if_data: z-стек xy-границ интер фолликулярных зон лимфоузла
        :param med_data: z-стек xy-границ медуллярной зоны лимфоузла
        """
        z_min = 0
        # # print(point, int(round(point[2] / 0.4) - z_min / 0.4))
        # scs_bord = scs_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        # if_zone = if_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        # med_zone = med_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        #
        # p = geom.Point(point[0], point[1])

        par_4 = 20 * 3 * 60 * 12
        if not self.act_time:
            self.act_time = 0

        self.activate = True

        self.curr_act_time = len(self.path)
        if not self.act_time:
            self.act_time = len(self.path)

        # разные сценарии для активированной и нет клетки
        # if self.activate:
        #     # изменения, связанные с активацией (скорость и тд)
        #     self.act_changes()
        #     if self.act_time:
        #         curr_time = (len(self.path) - self.act_time) * 20
        #         if curr_time >= par_4:
        #             if p.distance(if_zone[0]) == 5:
        #                 self.curr_act_time = len(self.path)
        #
        #     # проверка не находится ли клетка активированной слишком долго
        #     # если попала в IF, то активация
        #     if p.within(if_zone[0]) or p.within(if_zone[1]):
        #         self.activate = True
        #
        #         self.curr_act_time = len(self.path)
        #
        #     # если близко к медуллярной зоне, то активация
        #     elif p.distance(med_zone[0]) < 2:
        #         self.activate = True
        #         # self.set_state("active")
        #         self.curr_act_time = len(self.path)
        #         #  если близко к капсуле, то активация
        #     elif p.distance(scs_bord[1]) < 2:
        #         self.activate = True
        #         # self.set_state("active")
        #         self.curr_act_time = len(self.path)
        #
        #     if self.activate and len(self.path) - self.act_time >= self.act_max_time:
        #         self.activate = False
        #
        #         self.curr_act_time = None
        #         self.act_time = None
        #
        # else:
        #     # если не активирована, проверяем не попала ли куда
        #     # self.curr_act_time = len(self.path)
        #     # похоже на костыль но нужный
        #     if not self.act_time:
        #         self.act_time = 0
        #     # если попала в IF, то активация
        #     if p.within(if_zone[0]) or p.within(if_zone[1]):
        #         self.activate = True
        #
        #         self.curr_act_time = len(self.path)
        #         if not self.act_time:
        #             self.act_time = len(self.path)
        #     # если близко к медуллярной зоне, то активация
        #     elif p.distance(med_zone[0]) < 2:
        #         self.activate = True
        #         # self.set_state("active")
        #         self.curr_act_time = len(self.path)
        #         if not self.act_time:
        #             self.act_time = len(self.path)
        #             #  если близко к капсуле, то активация
        #     elif p.distance(scs_bord[1]) < 2:
        #         self.activate = True
        #         # self.set_state("active")
        #         self.curr_act_time = len(self.path)
        #         if not self.act_time:
        #             self.act_time = len(self.path)
        #     # else:
        #     #     self.set_state("inactive")

        # теперь нужна функция модуляции скорости и угла в зависимости от таймера и расположения внутри фолликулы

    def act_changes(self):
        """
        Функция изменения скорости при разных этапах активации
        """

        # ряд параметров которые означают различные временные промежутки жизни клетки после активации
        # время "замирания" сразу после активации, по идее от 2x до 30ти минут (последний множитель)
        # в промежутке между 1м и 2м клетки двигаются к scs
        par_1 = 20 * 3 * 15
        # после этого момента случайное блуждание пока не окажутся в 140 мкм от границы фолликулы
        par_2 = 20 * 3 * 180
        # время к которому они должны оказаться на границе фолликул, то есть в Т/В - зоне
        par_3 = 20 * 3 * 60 * 6

        par_4 = 20 * 3 * 60 * 12
        curr_time = (len(self.path) - self.act_time) * 20

        if curr_time < par_1:
            self.set_v_abs(0)
            print('0')
        # v_abs = 0
        elif par_1 <= curr_time < par_2:
            self.set_v_abs(0.046 * 0.5)
            print('1')
        elif par_2 <= curr_time < par_3:
            self.set_v_abs(0.046)
            print('2', end=',')

        elif par_4 <= curr_time:
            self.set_v_abs(0.046 * 1.5)
            print('\n see:', self.act_time, self.curr_act_time,
                  decision(self.exp_data[int((len(self.path) - self.curr_act_time) / 20)]))

            if decision(self.exp_data[int((len(self.path) - self.curr_act_time) / 20)]):
                self.activate = False
                self.act_time = None
                self.curr_act_time = None

            print('3')

        if self.activate and len(self.path) - self.act_time >= self.act_max_time:
            self.activate = False
            self.curr_act_time = None
            self.act_time = None

    # v_abs = стандартное значение, плюс модуляция угла в сторону сцс
    # появляется модуляция угла в сторону Т/В границы (по сути границы фолликулы)

    # Вопросы которые надо обсудить
    # 1. как модулируется угол, то что написано не подходит (либо я тупой)
    # так как сейчас это реализовано как урезание угла на какую-то величину (не понятно как её рассчитывать,
    # и если честно идей ноль, можно придумать что-тио с расчётом расстояния
    # до нужной границы, но что делать если клетка подошла вплотную,
    # я не могу родить математический расчёт для этого)

    # 2. норм ли будет каждый раз проверять все фолликулы на этом срезе и
    # что делать с вертикальной границей? (или лучше сделать отдельный параметр и проверять его в других модулях)

    # 3. как правильно поменять значение угла (плюс перепроверь как меняется это
    # значение, насколько я помню они не мгновенно ускоряются)

    # 4. нам нужно сделать "туннель" на границе и я не ебу как

    #  P.S. в моём понимании эта функция используется в конце шага и
    #  проверяет эти параметры накладывая ограничения и тд
    def curr_area(self, scs_data, if_data, med_data, red_data, blue_data, ext_red_data):
        """
        Функция для записи зоны нахождения в данные
        """

        if self.meet_state:
            return 'meet'

        point = self.path[-1]
        z_min = 0
        p = geom.Point(point[0], point[1])
        scs_zone = scs_data[int(round(point[2] / 0.4) - z_min / 0.4)][1]
        if_zone = if_data[int(round(point[2] / 0.4) - z_min / 0.4)][0]
        med_zone = med_data[int(round(point[2] / 0.4) - z_min / 0.4)][0]
        red_zone = red_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        ext_red_zone = ext_red_data[int(round(point[2] / 0.4) - z_min / 0.4)]
        _c = ''

        if p.within(if_zone):
            for fol in red_zone.keys():
                if p.within(red_zone[fol][0][0]):
                    # print(p.distance(red_zone[fol][1][0]))
                    if 140 * 0.4 < p.distance(red_zone[fol][1][0]) < 200 * 0.4:
                        _c = '_140_200_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]
                    elif 20 * 0.4 < p.distance(red_zone[fol][1][0]) < 80 * 0.4:
                        _c = '_20_80_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]

            return 'IF'
        elif p.distance(scs_zone) < 5:
            for fol in red_zone.keys():
                if p.within(red_zone[fol][0][0]):
                    # print(p.distance(red_zone[fol][1][0]))
                    if 140 * 0.4 < p.distance(red_zone[fol][1][0]) < 200 * 0.4:
                        _c = '_140_200_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]
                    elif 20 * 0.4 < p.distance(red_zone[fol][1][0]) < 80 * 0.4:
                        _c = '_20_80_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]
            return 'scs'
        elif p.within(med_zone):
            return 'med'
        elif p.within(ext_red_zone[0]):

            return 'ext_red'
        else:
            for fol in red_zone.keys():
                if p.within(red_zone[fol][0][0]):
                    # print(p.distance(red_zone[fol][1][0]))
                    if 140 * 0.4 < p.distance(red_zone[fol][1][0]) < 200 * 0.4:
                        _c = '_140_200_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]
                    elif 20 * 0.4 < p.distance(red_zone[fol][1][0]) < 80 * 0.4:
                        _c = '_20_80_' + fol + '_' + str(p.distance(red_zone[fol][1][0]))[:3]

                    return fol[:3]

            return 'ext_red'

    def meet_changes(self, act_check):
        # нужно понять надо ли выключать параметры связанные с активацией или можно только activate
        # Upd.: вроде норм поправил в act_changes
        if act_check:
            self.meet_state = True
            self.set_v_abs(0.046 * 0)
            self.activate = False
        else:
            pass
