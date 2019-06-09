"""
Computer Simulations of Sensory Systems, spring semester 2019
Exercise 3: The Vestibular System - Vestibular Implant Simulation

Authors: Vitaliy Banov, Savina Kim, Ephraim Seidenberg
Emails: vitanov@ruri.waseda.jp, savkim@ethz.ch, sephraim@ethz.ch

Date: June 9, 2019

Description:
    The vestibular system plays a critical role in our sensory system helping us stabilize our visual understanding and
    maintaining balance such as our head and body postures. As a result, this system is necessary to orient ourselves in
    space. The following program aims to simulate a vestibular implant according to the steps described in the task
    analysis and delivers the output as instructed.

"""

import task_analysis
from vestibular import *


def main():
    # process movement data
    cupular_deflection, otolithic_stimulation, head_orientations = task_analysis.move()
    # store the deflection data as instructed
    np.savetxt("CupularDisplacement.txt", cupular_deflection, fmt='%10.5f')
    # store the acceleration data as instructed
    np.savetxt("MaxAcceleration.txt", otolithic_stimulation, fmt='%10.5f')
    print("Maximum cupular displacement in mm and accelerations in m/s^2 stored in current folder.")
    # get the nose orientation from the head orientation data
    nose_orientation = get_nose_from_head_orientation(head_orientations[-1])
    # set a readable number format for printing
    np.set_printoptions(precision=3, suppress=True)
    print("Orientation of the nose at the end of the walk:", nose_orientation)


if __name__ == '__main__':
    main()
