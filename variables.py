from minimum import *
from math import *
from observer import Observer


class VariableStar:
    def __init__(self, name: str, pair: str, variability_type: str, period: float, epoch: float, amplitude_p: float,
                 amplitude_s: float, d_eclipse_prim: float, d_eclipse_sec: float, d_minimum_prim: float,
                 d_minimum_sec: float, mag0: float, a_pri: float, d_pri: float, g_pri: float, c_pri: float,
                 a_sin1: float, a_sin2: float, a_sin3: float, apsidal_movement_correction: float, sec_phase: float,
                 a_sec: float, d_sec: float, g_sec: float, c_sec: float, a_cos1: float, a_cos2: float, a_cos3: float,
                 lc_offset: float, in_prediction: float, in_alert: float):
        self.__name = name
        self.__pair = pair
        self.__variability_type = variability_type
        self.__period = period
        self.__epoch = epoch
        self.__amplitude_p = amplitude_p
        self.__amplitude_s = amplitude_s
        self.__d_eclipse_prim = d_eclipse_prim
        self.__d_eclipse_sec = d_eclipse_sec
        self.__d_minimum_prim = d_minimum_prim
        self.__d_minimum_sec = d_minimum_sec
        self.__mag0 = mag0
        self.__a_pri = a_pri
        self.__d_pri = d_pri
        self.__g_pri = g_pri
        self.__c_pri = c_pri
        self.__a_sin1 = a_sin1
        self.__a_sin2 = a_sin2
        self.__a_sin3 = a_sin3
        self.__apsidal_movement_correction = apsidal_movement_correction
        self.__sec_phase = sec_phase
        self.__a_sec = a_sec
        self.__d_sec = d_sec
        self.__g_sec = g_sec
        self.__c_sec = c_sec
        self.__a_cos1 = a_cos1
        self.__a_cos2 = a_cos2
        self.__a_cos3 = a_cos3
        self.__lc_offset = lc_offset
        if in_prediction:
            self.__in_prediction = True
        else:
            self.__in_prediction = False
        if in_alert:
            self.__alert = True
        else:
            self.__alert = False

    def __str__(self):
        return "{0} {1}".format(self.__name, self.__pair)

    def in_alert(self):
        return self.__alert

    def change_in_alert(self, new):
        self.__alert = new

    def in_prediction(self):
        return self.__in_prediction

    def change_in_prediction(self, new: bool):
        self.__in_prediction = new

    def change_pair(self, new):
        self.__pair = new

    def name(self):
        return self.__name

    def pair(self):
        return self.__pair

    def period(self):
        return self.__period

    def epoch(self):
        return self.__epoch

    def variability_type(self):
        return self.__variability_type

    def amplitude_p(self):
        return self.__amplitude_p

    def amplitude_s(self):
        return self.__amplitude_s

    def d_eclipse_prim(self):
        return self.__d_eclipse_prim

    def d_eclipse_sec(self):
        return self.__d_eclipse_sec

    def d_minimum_prim(self):
        return self.__d_minimum_prim

    def d_minimum_sec(self):
        return self.__d_minimum_sec

    def mag0(self):
        return self.__mag0

    def a_pri(self):
        return self.__a_pri

    def d_pri(self):
        return self.__d_pri

    def g_pri(self):
        return self.__g_pri

    def c_pri(self):
        return self.__c_pri

    def a_sin1(self):
        return self.__a_sin1

    def a_sin2(self):
        return self.__a_sin2

    def a_sin3(self):
        return self.__a_sin3

    def apsidal_movement_correction(self):
        return self.__apsidal_movement_correction

    def sec_phase(self):
        return self.__sec_phase

    def a_sec(self):
        return self.__a_sec

    def d_sec(self):
        return self.__d_sec

    def g_sec(self):
        return self.__g_sec

    def c_sec(self):
        return self.__c_sec

    def a_cos1(self):
        return self.__a_cos1

    def a_cos2(self):
        return self.__a_cos2

    def a_cos3(self):
        return self.__a_cos3

    def lc_offset(self):
        return self.__lc_offset

    def change_name(self, new):
        self.__name = new

    def change_period(self, new_value):
        self.__period = new_value

    def change_epoch(self, new_value):
        self.__epoch = new_value

    def change_variability_type(self, new_value):
        self.__variability_type = new_value

    def change_amplitude_p(self, new_value):
        self.__amplitude_p = new_value

    def change_amplitude_s(self, new_value):
        self.__amplitude_s = new_value

    def change_d_eclipse_prim(self, new_value):
        self.__d_eclipse_prim = new_value

    def change_d_eclipse_sec(self, new_value):
        self.__d_eclipse_sec = new_value

    def change_d_minimum_prim(self, new_value):
        self.__d_minimum_prim = new_value

    def change_d_minimum_sec(self, new_value):
        self.__d_minimum_sec = new_value

    def change_mag0(self, new_value):
        self.__mag0 = new_value

    def change_a_pri(self, new_value):
        self.__a_pri = new_value

    def change_d_pri(self, new_value):
        self.__d_pri = new_value

    def change_g_pri(self, new_value):
        self.__g_pri = new_value

    def change_c_pri(self, new_value):
        self.__c_pri = new_value

    def change_a_sin1(self, new_value):
        self.__a_sin1 = new_value

    def change_a_sin2(self, new_value):
        self.__a_sin2 = new_value

    def change_a_sin3(self, new_value):
        self.__a_sin3 = new_value

    def change_apsidal_movement_correction(self, new_value):
        self.__apsidal_movement_correction = new_value

    def change_sec_phase(self, new_value):
        self.__sec_phase = new_value

    def change_a_sec(self, new_value):
        self.__a_sec = new_value

    def change_d_sec(self, new_value):
        self.__d_sec = new_value

    def change_g_sec(self, new_value):
        self.__g_sec = new_value

    def change_c_sec(self, new_value):
        self.__c_sec = new_value

    def change_a_cos1(self, new_value):
        self.__a_cos1 = new_value

    def change_a_cos2(self, new_value):
        self.__a_cos2 = new_value

    def change_a_cos3(self, new_value):
        self.__a_cos3 = new_value

    def change_lc_offset(self, new_value):
        self.__lc_offset = new_value

    def first_minimum(self, jd):  # it´s return the first minimum after jd ( julian day )
        primary = True
        next_minimum = ceil((jd - self.__epoch) / self.__period) * self.__period + self.__epoch
        if next_minimum - self.__period * (1 - self.__sec_phase) >= jd:
            next_minimum = next_minimum - self.__period * (1 - self.__sec_phase)
            primary = False
        return Minimum(next_minimum, primary)

    def next_minimum(self, jd_last_min, primary):
        if primary:
            jd_next = jd_last_min + self.__period * self.__sec_phase
            primary_next = False
        else:
            jd_next = jd_last_min + self.__period * (1 - self.__sec_phase)
            primary_next = True
        return Minimum(jd_next, primary_next)

    def prediction(self, star: Star, sun: Sun, moon: Moon, obs_place: Observer, jd_start: float, jd_end: float,
                   visibility_sun=True, visibility_horizon=True, visibility_moon=False, moon_max_phase=100,
                   moon_min_distance=0):
        prediction = []
        if self.__period > 0 and self.__in_prediction:
            star_minimum = self.first_minimum(jd_start)
            while star_minimum.minimum_jd() < jd_end:
                visibility = True
                if visibility_sun and not star_minimum.visibility_sun(sun, obs_place.min_sunset, obs_place.latitude,
                                                                      obs_place.longitude):
                    visibility = False
                if visibility and visibility_horizon and not star_minimum.visibility_horizon(obs_place, star):
                    visibility = False

                if visibility and visibility_moon:
                    if not star_minimum.visibility_moon(moon, star, moon_max_phase, moon_min_distance,
                                                        obs_place.latitude, obs_place.longitude):
                        visibility = False
                if visibility:
                    prediction.append(star_minimum)
                star_minimum = self.next_minimum(star_minimum.minimum_jd(), star_minimum.minimum_type())
        if self.__period and self.__alert and self.__d_eclipse_prim:
            eclipse_duration_prim = 2 * self.__d_eclipse_prim
            eclipse_duration_sec = 2 * self.__d_eclipse_sec
            out_of_eclipse_prim = self.__period * self.__sec_phase - self.__d_eclipse_prim - self.__d_eclipse_sec
            out_of_eclipse_sec = self.__period * (1 - self.__sec_phase) - self.__d_eclipse_prim - self.__d_eclipse_sec
            first_prim_eclipse_start_before = (ceil((jd_start - self.__epoch + self.__d_eclipse_prim) / self.__period) - 1) \
                                              * self.__period + self.__epoch - self.__d_eclipse_prim
        return prediction

    def lightcurve(self, jd_start, jd_end, step: int, add_mag_and_offset=True, another_model=False, model=[],
                   time_points=[]):
        try:
            if another_model:
                mag0 = model[0]
                sec_ph = model[1]
                a_pri = model[2]
                d_pri = model[3]
                g_pri = model[4]
                c_pri = model[5]
                a_sec = model[6]
                d_sec = model[7]
                g_sec = model[8]
                c_sec = model[9]
                sin1 = model[10]
                sin2 = model[11]
                sin3 = model[12]
                cos1 = model[13]
                cos2 = model[14]
                cos3 = model[15]
                ap_c = model[16]
                period = model[17]
                epoch = self.__epoch + model[18]
            else:
                mag0 = self.__mag0
                sec_ph = self.__sec_phase
                a_pri = self.__a_pri
                d_pri = self.__d_pri
                g_pri = self.__g_pri
                c_pri = self.__c_pri
                a_sec = self.__a_sec
                d_sec = self.__d_sec
                g_sec = self.__g_sec
                c_sec = self.__c_sec
                sin1 = self.__a_sin1
                sin2 = self.__a_sin2
                sin3 = self.__a_sin3
                cos1 = self.__a_cos1
                cos2 = self.__a_cos2
                cos3 = self.__a_cos3
                ap_c = self.__apsidal_movement_correction
                period = self.__period
                epoch = self.__epoch
        except:
            mag0 = self.__mag0
            sec_ph = self.__sec_phase
            a_pri = self.__a_pri
            d_pri = self.__d_pri
            g_pri = self.__g_pri
            c_pri = self.__c_pri
            a_sec = self.__a_sec
            d_sec = self.__d_sec
            g_sec = self.__g_sec
            c_sec = self.__c_sec
            sin1 = self.__a_sin1
            sin2 = self.__a_sin2
            sin3 = self.__a_sin3
            cos1 = self.__a_cos1
            cos2 = self.__a_cos2
            cos3 = self.__a_cos3
            ap_c = self.__apsidal_movement_correction
            period = self.__period
            epoch = self.__epoch

        time_duration = jd_end - jd_start
        lightcurve_point = []
        try:
            if time_points:
                for jd in time_points:
                    phase = ((jd - epoch) / period) % 1
                    if phase >= 0.75:
                        phase = phase - 1
                    correction = ((jd - epoch) / period) * ap_c + sec_ph
                    pri = a_pri * (1 + c_pri * phase ** 2 / d_pri ** 2) * (
                                1 - ((1 - exp(1 - cosh(phase / d_pri))) ** g_pri))
                    sec = a_sec * (1 + c_sec * (phase - correction) ** 2 / d_sec ** 2) * (
                                1 - ((1 - exp(1 - cosh((phase - correction) / d_sec))) ** g_sec))
                    goniom1 = sin1 * sin(2 * pi * phase) + cos1 * cos(2 * pi * phase)
                    goniom2 = sin2 * sin(4 * pi * phase) + cos2 * cos(4 * pi * phase)
                    goniom3 = sin3 * sin(8 * pi * phase) + cos3 * cos(8 * pi * phase)
                    model_point = pri + sec + goniom1 + goniom2 + goniom3
                    if isnan(model_point):
                        return []
                    if add_mag_and_offset:
                        model_point = -model_point + mag0 + self.__lc_offset
                    lightcurve_point.append(model_point)
            else:
                for i in range(step):
                    jd = jd_start + i * time_duration/step
                    phase = ((jd - epoch) / period) % 1
                    if phase >= 0.75:
                        phase = phase - 1
                    correction = ((jd - epoch) / period) * ap_c + sec_ph
                    pri = a_pri * (1 + c_pri * phase ** 2 / d_pri ** 2) * (1 - ((1 - exp(1 - cosh(phase / d_pri))) ** g_pri))
                    sec = a_sec * (1 + c_sec * (phase - correction) ** 2 / d_sec ** 2) * (1 - ((1 - exp(1 - cosh((phase - correction) / d_sec))) ** g_sec))
                    goniom1 = sin1 * sin(2 * pi * phase) + cos1 * cos(2 * pi * phase)
                    goniom2 = sin2 * sin(4 * pi * phase) + cos2 * cos(4 * pi * phase)
                    goniom3 = sin3 * sin(8 * pi * phase) + cos3 * cos(8 * pi * phase)
                    model_point = pri + sec + goniom1 + goniom2 + goniom3
                    if isnan(model_point):
                        return []
                    if add_mag_and_offset:
                        model_point = -model_point + mag0 + self.__lc_offset
                    lightcurve_point.append(model_point)
        except:
            lightcurve_point = []
        return lightcurve_point

    def lightcurve_point(self, jd, another_model=False, model=[]):
        try:
            if another_model:
                sec_ph = model[1]
                a_pri = model[2]
                d_pri = model[3]
                g_pri = model[4]
                c_pri = model[5]
                a_sec = model[6]
                d_sec = model[7]
                g_sec = model[8]
                c_sec = model[9]
                sin1 = model[10]
                sin2 = model[11]
                sin3 = model[12]
                cos1 = model[13]
                cos2 = model[14]
                cos3 = model[15]
                ap_c = model[16]
                period = model[17]
                epoch = self.__epoch + model[18]
            else:
                sec_ph = self.__sec_phase
                a_pri = self.__a_pri
                d_pri = self.__d_pri
                g_pri = self.__g_pri
                c_pri = self.__c_pri
                a_sec = self.__a_sec
                d_sec = self.__d_sec
                g_sec = self.__g_sec
                c_sec = self.__c_sec
                sin1 = self.__a_sin1
                sin2 = self.__a_sin2
                sin3 = self.__a_sin3
                cos1 = self.__a_cos1
                cos2 = self.__a_cos2
                cos3 = self.__a_cos3
                ap_c = self.__apsidal_movement_correction
                period = self.__period
                epoch = self.__epoch
        except:
            sec_ph = self.__sec_phase
            a_pri = self.__a_pri
            d_pri = self.__d_pri
            g_pri = self.__g_pri
            c_pri = self.__c_pri
            a_sec = self.__a_sec
            d_sec = self.__d_sec
            g_sec = self.__g_sec
            c_sec = self.__c_sec
            sin1 = self.__a_sin1
            sin2 = self.__a_sin2
            sin3 = self.__a_sin3
            cos1 = self.__a_cos1
            cos2 = self.__a_cos2
            cos3 = self.__a_cos3
            ap_c = self.__apsidal_movement_correction
            period = self.__period
            epoch = self.__epoch

        try:
            phase = ((jd - epoch) / period) % 1
            if phase >= 0.75:
                phase = phase - 1
            correction = ((jd - epoch) / period) * ap_c + sec_ph
            if d_pri == 0 and d_sec == 0 :
                return None
            pri = a_pri * (1 + c_pri * phase ** 2 / d_pri ** 2) * (1 - ((1 - exp(1 - cosh(phase / d_pri))) ** g_pri))
            sec = a_sec * (1 + c_sec * (phase - correction) ** 2 / d_sec ** 2) * (1 - ((1 - exp(1 - cosh((phase - correction) / d_sec))) ** g_sec))
            gon_polynom1 = sin1 * sin(2 * pi * phase) + cos1 * cos(2 * pi * phase)
            gon_polynom2 = sin2 * sin(4 * pi * phase) + cos2 * cos(4 * pi * phase)
            gon_polynom3 = sin3 * sin(8 * pi * phase) + cos3 * cos(8 * pi * phase)
            model_point = pri + sec + gon_polynom1 + gon_polynom2 + gon_polynom3
        except:
            return None
        return model_point


    def variable_to_txt(self):
        string = []
        string.append(str(self.__name))
        string.append(str(self.__pair))
        string.append(str(self.__variability_type))
        string.append(str(self.__period))
        string.append(str(self.__epoch))
        string.append(str(self.__amplitude_p))
        string.append(str(self.__amplitude_s))
        string.append(str(self.__d_eclipse_prim))
        string.append(str(self.__d_eclipse_sec))
        string.append(str(self.__d_minimum_prim))
        string.append(str(self.__d_minimum_sec))
        string.append(str(self.__mag0))
        string.append(str(self.__a_pri))
        string.append(str(self.__d_pri))
        string.append(str(self.__g_pri))
        string.append(str(self.__c_pri))
        string.append(str(self.__a_sin1))
        string.append(str(self.__a_sin2))
        string.append(str(self.__a_sin3))
        string.append(str(self.__apsidal_movement_correction))
        string.append(str(self.__sec_phase))
        string.append(str(self.__a_sec))
        string.append(str(self.__d_sec))
        string.append(str(self.__g_sec))
        string.append(str(self.__c_sec))
        string.append(str(self.__a_cos1))
        string.append(str(self.__a_cos2))
        string.append(str(self.__a_cos3))
        string.append(str(self.__lc_offset))
        if self.__in_prediction:
            string.append("1")
        else:
            string.append("0")
        if self.__alert:
            string.append("1")
        else:
            string.append("0")
        return string

class Variables:
    def __init__(self, variables, variables_key):
        self.__variables = variables
        self.__variables_key = variables_key

    def __str__(self):
        return "Variable_list"

    @property
    def key(self):
        return self.__variables_key

    @property
    def variables(self):
        return self.__variables

    def choice_by_name(self, name: str):
        star_variability = [[], []]
        for variable in self.__variables:
            if variable.name() == name:
                star_variability[0].append(variable)
                star_variability[1].append(variable.pair())
        return star_variability

    def add_variable(self, variable: VariableStar):
        self.__variables.append(variable)

    def delete_variable(self, delete_variable):
        delete_variable_index = None
        for i, variable in enumerate(self.__variables):
            if delete_variable.name() == variable.name():
                if delete_variable.pair() == variable.pair():
                    delete_variable_index = i
                elif delete_variable.pair() < variable.pair():
                    variable.change_pair(chr(ord(variable.pair())-1))
                else:
                    pass
        del self.__variables[delete_variable_index]

        pass

    def change_variables_set(self, new_set):
        self.__variables = new_set

    def change_key_set(self, new_set):
        self.__variables_key = new_set

    def exist(self, variable):
        pass

    def check_aperture(self, aperture):
        pass

    def save_variable(self, new_variable):
        for i, variable in enumerate(self.__variables):
            if new_variable.name() == variable.name() and new_variable.pair() == variable.pair():
                self.__variables[i] = new_variable
                return

    def find_variable(self, name_star: str, pair: str):
        for variable in self.__variables:
            if name_star == variable.name() and pair == variable.pair():
                return variable
        return None
