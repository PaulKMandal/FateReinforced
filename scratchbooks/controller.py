import pystk


def control(aim_point, current_vel):

    action = pystk.Action()
    
    target_velocity = 30
    
    if aim_point[0] > .3 or aim_point[0] < -.3:
        target_velocity = 20
        
    if aim_point[0] > .5 or aim_point[0] < -.5:
        target_velocity = 15
        
    if aim_point[0] > .7 or aim_point[0] < -.7:
        target_velocity = 10
    
    if current_vel < target_velocity:
        action.acceleration = 1
        
    if current_vel > target_velocity:
        action.acceleration = 0
        action.brake = True

    if aim_point[0] > .2 or aim_point[0] < -.2:
        action.drift = True
        

    action.steer = aim_point[0]

    return action


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
