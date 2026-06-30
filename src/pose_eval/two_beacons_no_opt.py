# estimate pose from two beacon sightings without optimisation
# the algorithm estimates pose based on (d1, d2, h1) the distances from both beacons and heading to the
# first then uses heading to the second, h2, to choose between two possible pose estimates
# positions of each beacon j in world coords are (xj, yj)
# initially beacon distances and headings are assumed to be accurate
from math import sqrt, atan2, pi


from pose_eval.utils.angles import angular_distance, average_angle, normalize_angle


def estimate_pose(d1, h1, xb1, yb1, d2, h2, xb2, yb2):
    """ Return pose estimate (x, y, theta) given distances d1,d2 and headings h1,h2 to two beacons at positions (xb1, yb1) and (xb2, yb2)

    returns None if the readings are inconsistent, i.e. circles do not intersect or one is contained within the other
    """
    candidates = intersections(d1, xb1, yb1, d2, xb2, yb2)
    if candidates is None:
        return None

    (x1, y1), (x2, y2) = candidates
    # print(f"candidate positions: ({x1:.2f}, {y1:.2f}) and ({x2: .2f}, {y2:.2f})")



    #    beacons = ((d1, h1, xb1, yb2), (d2, h2, xb2, yb2))
    diffs = []
    # estimate the robot's orientation theta1 assuming it is at the first intersection
    theta1 = atan2(yb1 - y1, xb1 - x1) - h1
    # and the orientation if at the second beacon
    theta2 = atan2(yb2 - y1, xb2 - x1) - h2
    # and calculate the error
    diff1 = angular_distance(theta1, theta2)
    # take the average of these as the estimated heading given this position
    theta_1 = average_angle((theta1, theta2))
    # print(f"{theta1=:.3f}, {theta2=:.3f}, {theta_1=:.3f}, {diff1=:.3f}")

    # do the same assuming the robot is at the second intersection
    theta1 = atan2(yb1 - y2, xb1 - x2) - h1
    theta2 = atan2(yb2 - y2, xb2 - x2) - h2
    diff2 = angular_distance(theta1, theta2)
    theta_2 = average_angle((theta1, theta2))
    # print(f"{theta1=:.3f}, {theta2=:.3f}, {theta_2=:.3f}, {diff2=:.3f}")

    # choose the candidate with the smaller absolute error
    pose = (x1, y1, theta_1) if diff1 < diff2 else (x2, y2, theta_2)
    # print(f"estimated pose: ({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.3f})")

    return pose


def intersections(r1, x1, y1, r2, x2, y2):
    """ Return intersections ((xd, yd), (xe, ye)) for two circles radii r1 and r2 centres (x1, y1) and (x2, y2)
    or None is there are no intersections
    """
    d = sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))  # distance between beacons
    # print(f"distance between beacons: {d:.3f}")
    if d >= r1 + r2:  # circles too far apart
        print("no intersections: circles too far apart")
        return None
    elif d < abs(r1 - r2):
        print("no intersections: small circle contained within large one")
        return None

    # calculate a, the distance from centre of first circle P0 to line joining the two intersections P3 and P3'
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    # calculate h, half the distance between the two intersections
    h = sqrt(r1 ** 2 - a ** 2)
    # print(f'{a=:.3f} {h=:.3f}')

    # calculate coords of C = (xc, yc) the intersection of lines joining circle centres and circle intersections. This is
    # a/d along the way from P0 to P1
    xc = x1 + a / d * (x2 - x1)
    yc = y1 + a / d * (y2 - y1)
    # print(f"centre point C = ({xc:.3f}, {yc:.3f})")

    # calculate coord of the two intersections D and E. These are magnitude h and orthogonal to direction of P2 - P1
    xd = xc + h / d * (y1 - y2)
    yd = yc + h / d * (x2 - x1)
    # print(f"intersection D: ({xd:.3f} {yd:.3f})")

    xe = xc + h / d * (y2 - y1)
    ye = yc + h / d * (x1 - x2)
    # print(f"intersection E: ({xe:.3f} {ye:.3f})")

    return (xd, yd), (xe, ye)


def robot_polar_coords(xb, yb, pose):
    """Return (distance, heading) from robot with pose (x, y, theta) to object at position (xb, yb)
    """
    xr, yr, theta = pose
    d = sqrt((xr - xb) ** 2 + (yr - yb) ** 2)
    h = normalize_angle(atan2(yb - yr, xb - xr) - theta)
    return d, h


# def normalise_angle(angle):
#     """ normalise angle to be in (-pi, pi]
#     """
#     two_pi = 2 * pi
#     # reduce the angle
#     angle = angle % two_pi
#
#     # force it to be the positive remainder, so that 0 <= angle < 360
#     angle = (angle + two_pi) % two_pi
#
#     # force into the minimum absolute value residue class, so that -180 < angle <= 180
#     if angle > pi:
#         angle -= two_pi
#
#     return angle


if __name__ == "__main__":
    print("facing forward")
    pose = 3, 0, 0
    xb1, yb1 = 1, 0
    xb2, yb2 = -2, 5

    x, y, theta = pose
    print(f"true pose: ({x:.2f}, {y:.2f}, {theta:.3f})")
    print(f"beacon positions ({xb1:.2f}, {yb1:.2f}) ({xb2:.2f}, {yb2:.2f})")
    d1, h1 = robot_polar_coords(xb1, yb1, pose)
    print(f"{d1=:.3f} {h1=:.3f}")
    d2, h2 = robot_polar_coords(xb2, yb2, pose)
    print(f"{d2=:.3f} {h2=:.3f}")
    p = estimate_pose(d1, h1, xb1, yb1, d2, h2, xb2, yb2)
    if p is not None:
        xe, ye, thetae = p
        print(f"estimated pose: ({xe:.2f}, {ye:.2f}, {thetae:.3f})")
    else:
        print("cannot estimate pose fron inconsistent data")
    print()
    # introduce small error
    x, y, theta = pose
    print(f"true pose: ({x:.2f}, {y:.2f}, {theta:.3f})")
    print(f"beacon positions ({xb1:.2f}, {yb1:.2f}) ({xb2:.2f}, {yb2:.2f})")
    d1, h1 = robot_polar_coords(xb1, yb1, pose)
    h1 += 0.01
    print(f"{d1=:.3f} {h1=:.3f}")
    d2, h2 = robot_polar_coords(xb2, yb2, pose)
    h2 += -0.01
    print(f"{d2=:.3f} {h2=:.3f}")
    p = estimate_pose(d1, h1, xb1, yb1, d2, h2, xb2, yb2)
    if p is not None:
        xe, ye, thetae = p
        print(f"estimated pose: ({xe:.2f}, {ye:.2f}, {thetae:.3f})")
    else:
        print("cannot estimate pose fron inconsistent data")

    print()
    print("facing backward")
    pose = 3, 0, -pi
    xb1, yb1 = 1, 0
    xb2, yb2 = -2, 5

    x, y, theta = pose
    print(f"true pose: ({x:.2f}, {y:.2f}, {theta:.3f})")
    print(f"beacon positions ({xb1:.2f}, {yb1:.2f}) ({xb2:.2f}, {yb2:.2f})")
    d1, h1 = robot_polar_coords(xb1, yb1, pose)
    print(f"{d1=:.3f} {h1=:.3f}")
    d2, h2 = robot_polar_coords(xb2, yb2, pose)
    print(f"{d2=:.3f} {h2=:.3f}")
    p = estimate_pose(d1, h1, xb1, yb1, d2, h2, xb2, yb2)
    if p is not None:
        xe, ye, thetae = p
        print(f"estimated pose: ({xe:.2f}, {ye:.2f}, {thetae:.3f})")
    else:
        print("cannot estimate pose from inconsistent data")
    print()
    # introduce small error
    x, y, theta = pose
    print(f"true pose: ({x:.2f}, {y:.2f}, {theta:.3f})")
    print(f"beacon positions ({xb1:.2f}, {yb1:.2f}) ({xb2:.2f}, {yb2:.2f})")
    d1, h1 = robot_polar_coords(xb1, yb1, pose)
    h1 += 0.01
    print(f"{d1=:.3f} {h1=:.3f}")
    d2, h2 = robot_polar_coords(xb2, yb2, pose)
    h2 += -0.01
    print(f"{d2=:.3f} {h2=:.3f}")
    p = estimate_pose(d1, h1, xb1, yb1, d2, h2, xb2, yb2)
    if p is not None:
        xe, ye, thetae = p
        print(f"estimated pose: ({xe:.2f}, {ye:.2f}, {thetae:.3f})")
    else:
        print("cannot estimate pose fron inconsistent data")

