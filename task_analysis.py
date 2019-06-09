from vestibular import *

def move():
    """
    execute the steps to describe a walk according to the task analysis of the lecture
    :return: maximum deflection of the cupula and stimuli on otolith organ as well as orientations of the head
    """
    # step 1 of task analysis: get data
    data = get_data('MovementData/Walking_02.txt')
    # step 2: get the initial orientation of the sensor
    sensor_orientation = get_init_orientation_sensor(data.acc[0])
    # step 3: get the vector of the right horizontal semi-circular canal's on-direction
    rhscc_init_on_dir = get_init_on_dir_rh_scc(15)
    # preparation for step 4: align the angular velocity sensor data with the global coordinate system
    angular_velocities_aligned_globally = align_sensor_data_globally(data.omega, sensor_orientation)
    # step 4: calculate the stimulation of the cupula
    stimuli = get_scc_stimulation(angular_velocities_aligned_globally, rhscc_init_on_dir)
    # step 5: get the transfer function of the scc with the dynamics provided in the lecture
    scc_trans_fun = get_scc_transfer_fun(0.01, 5)
    # step 6: get the cupular deflection
    max_cupular_deflection = calculate_max_cupular_deflection(scc_trans_fun, stimuli, data.rate)
    # preparation for step 7: align the acceleration sensor data with the global coordinate system
    accelerations_aligned_globally = align_sensor_data_globally(data.acc, sensor_orientation)
    # step 8: calculate the maxmimum left- and rightwards stimulation of the otolithic organ
    max_left_right_stimuli = calculate_otolithic_max_stimuli(accelerations_aligned_globally, 1)
    # step 9: calculate the head orientation
    head_orientations = calculate_head_orientation(angular_velocities_aligned_globally, data.rate)

    return max_cupular_deflection, max_left_right_stimuli, head_orientations
