import numpy as np
from skinematics.sensors.xsens import XSens
from skinematics import vector
from scipy.constants import g
from numpy import sin, pi


def main():
    # declare input file name
    in_file = 'Walking_01.txt'
    # TODO: Catch case where sensor data is not in the same folder (e.g. open file dialog)
    # read in data
    data = XSens(in_file, q_type=None)
    # process movement data
    print(move(data))


def get_init_orientation_sensor(resting_acc_vector):
    """
    Calculates the approximate orientation of the gravity vector of an inertial measurement unit (IMU), assuming a
    resting state and with respect to a global, space-fixed coordinate system.
    Underlying idea: Upwards movements should be in positive y-direction, so the provided acceleration vector is aligned
    with a vector with the gravitational constant as it's length lying on the y-axis and pointing in positive direction.
    :param resting_acc_vector: acceleration vector from imu, should be pointing upwards, i.e. measured at rest.
    :return: a vector around which a rotation of the sensor would approximately align it's edges with the global
    coordinate system
    """
    # create a vector with the gravitational constant as it's length in positive y-axis direction
    up_direction_global = np.array([0, g, 0])
    # determine the vector around which a rotation of the sensor would approximately align it's edges with a global,
    # space fixed coordinate system
    sensor_orientation_approx = vector.q_shortest_rotation(up_direction_global, resting_acc_vector)

    return sensor_orientation_approx


def adjust_sensor_data(sensor_data: XSens, global_orientation):
    """"""
    adjusted_data = vector.rotate_vector(sensor_data, global_orientation)

    return adjusted_data


def get_init_on_dir_rh_scc(reid_initial_pitch_angle_deg):
    """

    :param reid_initial_pitch_angle_deg: the angle about which Reid's line is pitched upwards in degrees, with respect
    to fixed space horizontal plane
    :return:
    """
    # set the vector about which rotations stimulate the right horizontal semi-circular canal respective to the
    # orientation of Reid's line given by the exercise and normalize it
    on_dir = vector.normalize(np.array([0.32269, -0.03837, -0.94573]))
    # convert the initial pitch angle to radian
    reid_initial_pitch_angle_rad = reid_initial_pitch_angle_deg / 180 * pi
    # set the rotation vector for the pitch of Reid's line with the defined angle
    reid_initial_pitch = np.array([0, 0, sin(reid_initial_pitch_angle_rad/2)])
    # adjust the on direction vector of the semi-circular canal according to the pitch of Reid's line so that it now
    # represents the orientation in the global coordinate system
    on_dir_adjusted = vector.rotate_vector(on_dir, reid_initial_pitch)

    return on_dir_adjusted


def get_cupular_stimulation(angular_velocity_data, scc_orientation):
    return angular_velocity_data @ scc_orientation


def move(data: XSens):
    # get the initial orientation of the sensor
    sensor_orientation = get_init_orientation_sensor(data.acc[0])
    # align the sensor data with the global coordinate system
    angular_velocities_adjusted = adjust_sensor_data(data.omega, sensor_orientation)
    # get the vector of the right horizontal semi-circular canal's on-direction
    rhscc_init_on_dir = get_init_on_dir_rh_scc(15)
    # calculate the stimulation of the cupula
    return get_cupular_stimulation(angular_velocities_adjusted, rhscc_init_on_dir)


if __name__ == 'main':
    main()