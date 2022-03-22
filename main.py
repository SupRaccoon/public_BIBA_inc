import argparse
import datetime
import itertools
import os
import pickle
import time
import random

import numpy as np
from loguru import logger
from scipy.spatial.distance import pdist, cdist
from tqdm.auto import tqdm

from cells import cell


def setup_args():
    parser = argparse.ArgumentParser(description="Computing cells movement")

    parser.add_argument('-iters', '--iterations',
                        action='store', help="Number of iterations for cells",
                        type=int, required=True)
    parser.add_argument('-B_nc', '--B_num_cells',
                        action='store', help="Number of B-cells",
                        type=int, required=True)
    parser.add_argument('-T_nc', '--T_num_cells',
                        action='store', help="Number of T-cells",
                        type=int, required=True)
    parser.add_argument('-data_path', '--data_path',
                        action='store', help="Path to the folder with map data",
                        type=str, required=False)
    parser.add_argument('-out', '--output',
                        action='store', help="Path for output data",
                        type=str, required=False)
    parser.add_argument('-mod_vel', '--mod_velocity',
                        action='store', help="Modulate velocity and path to the values",
                        type=str, required=False)
    parser.add_argument('-mod_B_coord', '--mod_B_start_point',
                        action='store', help="Modulate initial coordinate",
                        type=str, required=False)
    parser.add_argument('-mod_T_coord', '--mod_T_start_point',
                        action='store', help="Modulate initial coordinate",
                        type=str, required=False)
    parser.add_argument('-ss', '--start_steps',
                        action='store_true', help="Make random steps before the calculation starts",
                        required=False)
    return parser.parse_args()


class MyError(Exception):
    def __init__(self):
        self.txt = 'text'


def make_calculations(my_cell, iter_number=90, start_steps=False, cell_dep_start_steps=6):
    """
    Функция для проведения вычислений в создаваемых потоках. Принимает на вход клетку, делает вычисления в ней,
    затем возвращает её.
    :param my_cell: объект класса BasicCell или его дочернего класса. Клетка, для которой производятся расчеты
    :param iter_number: количество итераций расчетов, необходимое произвести с точкой.
    :param start_steps: нужно ли перед итерациями сделать несколько стартовых шагов.
    :param cell_dep_start_steps: максимальное количество стартовых шагов, выполняемых клеткой.
    :return: cell.BasicCell или его дочерний класс с результатами расчета.
    """
    try:
        if start_steps:
            num_of_steps = np.random.randint(0, cell_dep_start_steps)
            my_cell.movement(step_amount=num_of_steps)
        for _ in range(iter_number - 1):
            my_cell.movement()
        return my_cell
    except ValueError as ve_err:
        print(ve_err)
        return my_cell


def make_calculations_alt_ver(step_cell):
    step_cell.pre_step()


def check_distance(cell_pool, cell_ind, val=0.2):
    pool_coord = np.array([x.pre_point for x in cell_pool])
    v_dist = pdist(pool_coord)

    return cell_ind[np.where(v_dist < val)], v_dist


def check_meet(cell_pool_1, cell_pool_2, val=0.2):
    """
    Ох ебать, тут цирк с конями, кароч,на вход 2 массива, пол ним создаём 3хмерный массив парных индексов
    потом считаем дистанции -- получаем массив той же размерности что и массив индексов
    затем идём по массиву индексов чтобы корректно отработала функция where в результате массив всех индексов встреч
    :param cell_pool_1:
    :param cell_pool_2:
    :param val:
    :return:
    """
    cell_ind = np.array([[[i, j] for j in range(len(cell_pool_2))] for i in range(len(cell_pool_1))])
    pool_coord_1 = np.array([x.path[-1] for x in cell_pool_1])
    pool_coord_2 = np.array([x.path[-1] for x in cell_pool_2])
    v_dist = cdist(pool_coord_1, pool_coord_2)
    res = np.empty(shape=[0, 2])
    for i in range(len(cell_ind)):
        res = np.append(res, cell_ind[i][np.where(v_dist[i] < val)], axis=0)
    return res, v_dist


def change_vect_vel(step_cell, val=30):
    new_phi, new_theta = step_cell.rotation(val)
    step_cell.phi.set_index(new_phi)
    step_cell.theta.set_index(new_theta)


def back_angle(step_cell):
    step_cell.phi.set_index(step_cell.orig_phi)
    step_cell.theta.set_index(step_cell.orig_theta)


def cell_meet(cells_array, pair_indexes, distance=1.5):
    rq_deep = 0

    calc_final = False
    while not calc_final:
        if rq_deep < 1000:
            rq_deep += 1

            for curr_cell in cells_array:
                make_calculations_alt_ver(curr_cell)
            # проверка расстояний
            bad_pool, dist = check_distance(cells_array, pair_indexes, distance)
            if len(bad_pool) > 0:
                # смена направления
                for pair in bad_pool:
                    if random.random() < 0.5:
                        logger.trace(
                            f"{cells_array[pair[0]].cell_type}:{cells_array[pair[0]].cell_id} > " +
                            f"{len(cells_array[pair[0]].path)}: {cells_array[pair[0]].phi.index}, {cells_array[pair[0]].theta.index} ; change_1_angle_check")
                        change_vect_vel(cells_array[pair[0]])
                    else:
                        logger.trace(
                            f"{cells_array[pair[1]].cell_type}:{cells_array[pair[1]].cell_id}> " +
                            f"{len(cells_array[pair[0]].path)}: {cells_array[pair[1]].phi.index}, {cells_array[pair[1]].theta.index} ; change_2_angle_check")
                        change_vect_vel(cells_array[pair[1]])
                p_2 = []
                for curr_cell in cells_array:
                    p_2.append(curr_cell.v_dir)
                    make_calculations_alt_ver(curr_cell)
                # проверка
                bad_pool_2, dist_2 = check_distance(cells_array, pair_indexes, distance)
                if len(bad_pool_2) > 0:
                    # в случае неудачи возврат к старым углам
                    for pair in bad_pool:
                        back_angle(cells_array[pair[0]])
                        back_angle(cells_array[pair[1]])
                        logger.trace(
                            f"{cells_array[pair[0]].cell_type}:{cells_array[pair[0]].cell_id} and {cells_array[pair[1]].cell_id}> " +
                            f"{len(cells_array[pair[0]].path)}: {bad_pool_2, sorted(dist_2)[:4]} ; bad_angle_check")
                else:
                    calc_final = True
                    return calc_final
            else:
                calc_final = True
                return calc_final
        elif rq_deep == 1000 and distance > 0:

            rq_deep = 0
            distance -= 0.5
            logger.trace(
                f"{cells_array[0].cell_type}:{cells_array[0].cell_id}, {calc_final, rq_deep} ; bad_meet_check_down_dist_to_{distance}")
        else:
            calc_final = True
            logger.warning(
                f"{cells_array[0].cell_type}:{cells_array[0].cell_id}, {calc_final, rq_deep} ; bad_meet_check_down_dist_to_{distance}")
            return calc_final


if __name__ == "__main__":

    # ещё до аргументов инициализируем логирование
    logger.add(sink=f"logs/{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}.log",
               level="TRACE",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} - {message}",
               diagnose=False, colorize=True, enqueue=True, rotation="1 GB", encoding="utf-8")

    # парсим аргументы
    args = setup_args()
    for arg, value in sorted(vars(args).items()):
        logger.info(f"Argument {arg}: {value}")
    # получаем путь к данным
    DATA_PATH = args.data_path or "data_map/"
    logger.info(f"DataMap from {DATA_PATH}")
    # а затем считываем их в data_map
    data_map = dict()
    for name in os.listdir(DATA_PATH):
        with open(DATA_PATH + name, 'rb') as f:
            data_map[name] = pickle.load(f)

    # если задана вариация скоростей и/или координат, то присваиваем соответствующие значения
    # а именно считываем данные для вариации скоростей и/или координат, а затем задаем функцию,
    # которая для каждой клетки либо присваивает случайную скорость/координату из начальных данных,
    # либо возвращает None, тем самым задавая для клетки значения по умолчанию
    if args.mod_velocity:
        logger.info(f"Velocities modulation from {args.mod_velocity}")
        velocities = np.fromfile(args.mod_velocity, sep=",")
        velocities = velocities[np.where((velocities >= 0.032) & (velocities <= 0.2))]

        def declared_velocity():
            return np.random.choice(velocities)
    else:
        logger.info("Standard velocities")

        def declared_velocity():
            return None
    if args.mod_B_start_point:
        logger.info(f"Initial coordinates modulation from {args.mod_B_start_point}")
        B_coordinates = np.fromfile(args.mod_B_start_point, sep=",").reshape((-1, 3))
        print(len(B_coordinates))

        # здесь приходится делать random.choice из индексов coordinates, так как
        # random.choice не умеет в выбор из многомерного массива
        def declared_B_coordinate():
            return B_coordinates[np.random.choice(len(B_coordinates))]

    else:
        logger.info("Standard coordinates")

        def declared_B_coordinate():
            return None

    if args.mod_T_start_point:
        logger.info(f"Initial coordinates modulation from {args.mod_T_start_point}")

        T_coordinates = np.fromfile(args.mod_T_start_point, sep=",").reshape((-1, 3))
        print(len(T_coordinates))

        # здесь приходится делать random.choice из индексов coordinates, так как
        # random.choice не умеет в выбор из многомерного массива
        def declared_T_coordinate():
            return T_coordinates[np.random.choice(len(T_coordinates))]
    else:
        logger.info("Standard coordinates")

        def declared_T_coordinate():
            return None
    # создаем массив из num_cells клеток, каждой из них передаём data_map в качестве карты,
    # # а также функции для генерации начальных координат и скоростей
    B_cell_array = [cell.BCell(b_cell_init_coords=declared_B_coordinate(),
                               data_map=data_map) for _ in range(args.B_num_cells)]

    T_cell_array = [cell.TCell(t_cell_init_coords=declared_T_coordinate(),
                               data_map=data_map) for _ in range(args.T_num_cells)]

    B_pair_indexes = np.array(list(itertools.combinations(range(args.B_num_cells), 2)))
    T_pair_indexes = np.array(list(itertools.combinations(range(args.T_num_cells), 2)))
    # индексы для встречи Т и В клеток первый элемент -- номер В - клетки , второй -- номер Т-клетки
    # при этом все клетки объединяются в один массив [В, Т]
    B_T_pair_indexes = np.array([[i, j] for i in range(args.B_num_cells) for j in range(args.B_num_cells, args.T_num_cells)])
    # начинаем таймер, чтобы узнать, сколько времени уйдет
    logger.success("Calculation start")
    start_time = time.time()
    # TODO: нужно изменить все переменные на адекватные и добавить логирование для важных шагов
    # TODO: а ещё нужно разобраться с tqdm, нужен ли он нам, если да, то как используем
    # TODO: ДОБАВИТЬ ДОП СТАРТОВЫЕ ТОЧКИ ДЛЯ Т-клеток (сделано)
    # TODO: проверить столкновения и добавить остановку после него (сделано)
    # TODO: разделить один парметр на 2, а именно, инактивация и освобождение от Т/В зоны
    # try:
    meeting_pool = dict()
    meeting_pool["B"] = set()
    meeting_pool["T"] = set()

    for i in tqdm(range(args.iterations)):

        if cell_meet(B_cell_array, B_pair_indexes):
            if len(T_cell_array) > 0:
                if cell_meet(T_cell_array, T_pair_indexes):

                    for cell in B_cell_array:
                        logger.debug(f"{cell.cell_type}:{cell.cell_id} > " +
                                     f"{len(cell.path)}: {cell.phi.index}, {cell.theta.index}; BAS {len(cell.bad_angle_set)}")
                        cell.movement()
                    for cell in T_cell_array:
                        logger.debug(f"{cell.cell_type}:{cell.cell_id} > " +
                                     f"{len(cell.path)}: {cell.phi.index}, {cell.theta.index}; BAS {len(cell.bad_angle_set)}")
                        cell.movement()

                    meet_ind, pool_dist = check_meet(B_cell_array, T_cell_array, 5)

                    if len(meet_ind) > 0:
                        curr_B_pool = set(meet_ind.T[0]).difference(meeting_pool["B"])
                        curr_T_pool = set(meet_ind.T[1]).difference(meeting_pool["T"])
                        if len(curr_B_pool) > 0 and len(curr_T_pool) > 0:
                            for curr_pair in meet_ind:
                                cell_pair = [int(curr_pair[0]), int(curr_pair[1])]
                                if cell_pair[0] in curr_B_pool and cell_pair[1] in curr_T_pool:
                                    is_act = B_cell_array[cell_pair[0]].meet_changes()
                                    T_cell_array[cell_pair[1]].meet_changes(is_act)
                                    if is_act:
                                        meeting_pool["B"].add(curr_pair[0])
                                        meeting_pool["T"].add(curr_pair[1])
                                        # logger.debug(f"MEETING {meet_ind} {pool_dist} ")
                                        logger.debug(f"{B_cell_array[cell_pair[0]].cell_type} and {T_cell_array[cell_pair[1]].cell_type}:{B_cell_array[cell_pair[0]].cell_id} and {T_cell_array[cell_pair[1]].cell_id}> " +
                                                     f"{len(B_cell_array[cell_pair[0]].path)}: {pool_dist[cell_pair[0]][cell_pair[1]]} ; meeting")

        else:
            break
    # except MyError as ind_err:
    #     print(f"Index Error occurred")
    #     prefix = "w_IND_ERR"
    # else:
    prefix = ""

    logger.success(f"Calculation finished in {round(time.time() - start_time, 4)} seconds")

    # теперь запишем результаты
    if args.output:
        OUTPUT_PATH = args.output
    else:
        if os.path.exists("calc_output"):
            OUTPUT_PATH = "calc_output"
        else:
            OUTPUT_PATH = os.getcwd()
    logger.info(f"Output path {os.path.abspath(OUTPUT_PATH)}")
    # пока результат записывается в виде бинарного файла, чтобы сохранять именно объекты клеток в том виде,
    # в каком они находятся непосредственно после расчетов
    for sub_cell in B_cell_array:
        sub_cell.write_path(os.path.normpath(os.path.join(OUTPUT_PATH,
                                                          f"paths/{sub_cell.cell_type}/" +
                                                          f"{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}_{prefix}_{sub_cell.cell_id}.csv")))
        sub_cell.write_data(os.path.normpath(os.path.join(OUTPUT_PATH,
                                                          f"data/{sub_cell.cell_type}/" +
                                                          f"_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}_{sub_cell.cell_id}.csv")))
    for sub_cell in T_cell_array:
        sub_cell.write_path(os.path.normpath(os.path.join(OUTPUT_PATH,
                                                          f"paths/{sub_cell.cell_type}/" +
                                                          f"{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}_{prefix}_{sub_cell.cell_id}.csv")))
        sub_cell.write_data(os.path.normpath(os.path.join(OUTPUT_PATH,
                                                          f"data/{sub_cell.cell_type}/" +
                                                          f"_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}_{sub_cell.cell_id}.csv")))
