import numpy as np
from pathlib import Path
from tkinter import filedialog
from skinematics.sensors.xsens import XSens
from skinematics import vector
from skinematics import quat
from scipy import signal
from scipy.constants import g


def get_data(file_path):
    """
    Fetch data from provided path if valid and open a dialog window for path selection otherwise
    :param file_path: path to the file with sensor data
    :return: the data parsed as an XSense object
    """
    in_file = file_path
    # if no valid path was provided ...
    if not Path(in_file).exists():
        # ... get a valid path
        in_file = filedialog.askopenfilename(title="File not found in current folder. Please select file")
    # read in data
    data = XSens(in_file, q_type=None)

    return data


def get_init_orientation_sensor(resting_acc_vector):
    """
    Calculates the approximate orientation of the gravity vector of an inertial measurement unit (IMU), assuming a
    resting state and with respect to a global, space-fixed coordinate system.
    Underlying idea: Upwards movements should be in positive z-direction, but in the experimental setting, the sensor's
    z-axis is pointing to the right, so the provided acceleration vector is first aligned with a vector with the
    gravitational constant as it's length lying on the sensor y-axis and pointing in positive direction and then rolled
    90 degrees to the right along the x-axis
    :param resting_acc_vector: acceleration vector from imu, should be pointing upwards, i.e. measured at rest.
    :return: a vector around which a rotation of the sensor approximately aligns it's coordinate system with the global
    """
    # create a vector with the gravitational constant as it's length in positive y-axis direction
    g_vector = np.array([0, g, 0])
    # determine the vector around which a rotation of the sensor would align it's own (not global) y-axis with gravity
    sensor_orientation_g_aligned = vector.q_shortest_rotation(resting_acc_vector, g_vector)
    # create a rotation vector for a 90 degree rotation along the x-axis
    roll_right = vector.q_shortest_rotation(np.array([0, 1, 0]), np.array([0, 0, 1]))
    # roll the sensor coordinate system to the right
    sensor_orientation_global = quat.q_mult(roll_right, sensor_orientation_g_aligned)

    return sensor_orientation_global


def align_sensor_data_globally(sensor_data: XSens, global_orientation):
    """
    align sensor data with a global coordinate system
    :param sensor_data: the data to be adjusted
    :param global_orientation: the orientation of the sensor in the global coordinate system
    :return: the adjusted sensor data
    """
    # rotate the data the same way the sensor is rotated with respect to the global reference coordinate system
    adjusted_data = vector.rotate_vector(sensor_data, global_orientation)

    return adjusted_data


def get_init_on_dir_rh_scc(reid_initial_pitch_angle_deg):
    """
    get the initial orientation of the right horizontal semi-circular canal's on direction vector
    :param reid_initial_pitch_angle_deg: the angle about which Reid's line is pitched upwards in degrees, with respect
    to fixed space horizontal plane
    :return:
    """
    # set the vector about which rotations stimulate the right horizontal semi-circular canal respective to the
    # orientation of Reid's line given by the exercise and normalize it
    on_dir = vector.normalize(np.array([0.32269, -0.03837, -0.94573]))
    # convert the initial pitch angle to radian
    reid_initial_pitch_angle_rad = reid_initial_pitch_angle_deg / 180 * np.pi
    # set the rotation vector for the pitch of Reid's line with the defined angle
    reid_initial_pitch = np.array([0, 0, np.sin(reid_initial_pitch_angle_rad/2)])
    # adjust the on direction vector of the semi-circular canal according to the pitch of Reid's line so that it now
    # represents the orientation in the global coordinate system
    on_dir_adjusted = vector.rotate_vector(on_dir, reid_initial_pitch)

    return on_dir_adjusted


def get_scc_stimulation(angular_velocity_data, scc_on_direction):
    """
    calculate the stimulation of the semi-circular canal
    :param angular_velocity_data: the angular movement data
    :param scc_on_direction: a vector along which a rotation leads to stimulation of the semi-circular canal
    :return: the stimuli evoked by the angular movement
    """
    stimuli = angular_velocity_data @ scc_on_direction

    return stimuli

def get_scc_transfer_fun(T1, T2):
    """
    initialize a linear time invariant transfer function system for simulations of the semi-circular canal
    :param T1: time constant 1
    :param T2: time constant 2
    :return: the linear time invariant transfer function system
    """
    numerator = [T1 * T2, 0]
    denominator = [T1 * T2, T1 + T2, 1]
    # create the linear time invariant transfer function system
    trans_fun_sys = signal.lti(numerator, denominator)

    return trans_fun_sys


def calculate_max_cupular_deflection(scc_trans_fun_sys, stimuli, frequency):
    """
    Calculates the maximum deflection of the cupula due to given stimuli
    :param scc_trans_fun_sys: the transfer function for the semi-circular canal
    :param stimuli: the velocities exerted on the semi-circular canal
    :param frequency: sample rate of the sensor
    :return: the distances of highest deflection of the cupula evoked by the stimuli
    """
    # set the radius of the right semicircular canal in millimeters
    radius_rhscc = 3.2
    # set the time axis
    time_axis = np.arange(len(stimuli)) / frequency
    # simulate the response of the scc, i.e. calculate the output of the linear system
    _, out_signal, _ = signal.lsim(scc_trans_fun_sys, stimuli, time_axis)
    # multiply by the radius of the semicircular canal
    deflection = out_signal * radius_rhscc
    # calculate minimum and maximum discplacements
    deflection_extrema = np.array([np.min(deflection), np.max(deflection)])

    return deflection_extrema


def calculate_otolithic_max_stimuli(acceleration_data, axis=1):
    """
    Calculates the maximum stimulation exerted on the otolithic organ in both directions of a given axis
    :param acceleration_data: acceleration data from a sensor, adjusted to global coordinate system
    :param axis: 0 = x, 1 = y, 2 = z
    :return: the maximum stimuli in positive and negative direction on the selected axis
    """
    # get only the stimuli on the provided axis
    axial_stimuli = np.array([acceleration[axis] for acceleration in acceleration_data])
    # get the maximum stimuli in positive and negative direction
    axial_stimuli_extrema = np.array([np.min(axial_stimuli), np.max(axial_stimuli)])

    return axial_stimuli_extrema


def calculate_head_orientation(angular_velocity_data, frequency):
    """
    calculates the orientation of the head according to given angular velocities it undergoes as measured by a sensor,
    with the assumption that the head is perfectly aligned with the global coordinate system in the beginning
    :param angular_velocity_data: angular velocities tracked by a sensor
    :param frequency: sample rate of the sensor
    :return: the orientations of the head at each point in time
    """
    # calculate the quaternions describing the position of the head according to the angular velocities it underwent.
    head_orientation = quat.calc_quat(angular_velocity_data, [0, 0, 0], rate=frequency, CStype='bf')

    return head_orientation


def get_nose_from_head_orientation(head_orientation):
    """
    Get a vector describing the orientation of the noses end according to the position of the head
    :param head_orientation: the orientation of the head in the global coordinate system
    :return: the vector describing the nose orientation
    """
    # define the nose as pointing in x-direction initially
    nose_vector = np.array([1, 0, 0])
    # move the nose according to how the head is positioned
    nose_orientation = vector.rotate_vector(nose_vector, head_orientation)

    return nose_orientation
