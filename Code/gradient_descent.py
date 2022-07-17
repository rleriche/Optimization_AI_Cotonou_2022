
def linesearch(
    position: np.array,
    func: object,
    direction: np.array,
    LB: np.array,
    UB: np.array,
    maxloop=100,
    suffDecFact=0.1,
    decFact=0.5,
    initStepFact=1,
):

    """
    position : current point
    direction : direction in which to search.
    suffDecFact : scalar in [0,1[  used in the sufficient decrease test,
      i.e. how much better than Taylor is demanded.
    decFact : scalar in ]0,1[ , used to reduce stepSize
    initStepFact : multiplicative factor of gradient norm to determine initial stepSize
    LB,UB : d-dimensional vectors of lower and upper bounds for the variables
    """
    if len(LB) != len(position) or len(UB) != len(position):
        raise ValueError("the bounds should have %s components" % len(position))

    if not 0 <= suffDecFact < 1:
        raise ValueError(
            "the sufficient decrease factor (suffDecFact) should be between 0 and (strictly) 1"
        )

    gradf = get_gradient(func=func, x=position)
    func_value_at_position = func(x)
    size_of_domain = np.linalg.norm(UB - LB)
    normGrad = np.linalg.norm(gradf)

    stepSize = max(initStepFact * normGrad, (size_of_domain / 100))
    gradient_projected_on_direction = direction.dot(gradf)

    if gradient_projected_on_direction > 0:
        direction = -direction
        gradient_projected_on_direction = -gradient_projected_on_direction

    n_loop = 0

    condition = False

    while not condition:
        next_position = position + stepSize * direction

        lower_bound_violated = LB > next_position
        upper_bound_violated = next_position < UB

        # coordinates should be inside the domaine
        next_position_inbounds = (
            upper_bound_violated * UB
            + (1 - lower_bound_violated) * (1 - upper_bound_violated) * next_position
            + lower_bound_violated * LB
        )

        if np.linalg.norm(next_position_inbounds - next_position) < 10e-10:
            func_value_at_position = func(next_position_in_bound)
            n_loop += 1
        stepSize = decFact * stepSize

        condition = n_loop > maxloop or func_value_at_position < (
            func(position) + suffDecFact * stepSize * np.vdot(gradient_f_i.T, direction)
        )

    return stepSize


def get_gradient(func: object, x: np.array, epsilon: float = 1e-3) -> np.array:
    dimension = len(x)
    gradient = np.zeros(dimension)
    for coordinate_index in range(dimension):
        h = np.zeros(dimension)
        h[coordinate_index] = epsilon
        f_xh = func(x + h)
        f_x = func(x)
        gradient[coordinate_index] = (f_xh - f_x) / epsilon
    return gradient




def get_optima_steepest_descent(
    func: object,
    start_position: np.array,
    LB,
    UB,
    learning_rate: float = 1e-1,
    linesearch_type=False,
    min_step_size: float = 1e-5,
    min_grad_size: float = 1e-5,
    n_iterations: int = 1e3,
    max_double: float = 1e12,
):
    iteration = 0
    func_value_best_so_far = max_double
    position_best_so_far = start_position
    current_position = start_position
    condition = False
    while not condition:
        func_value_at_current_position = func(current_position)
        if func_value_at_current_position < func_value_best_so_far:
            position_best_so_far = current_position
            func_value_best_so_far = func_value_at_current_position
        gradient_at_current_position = get_gradient(func, current_position)
        gradient_size = np.linalg.norm(gradient_at_current_position)
        direction_at_current_position = -gradient_at_current_position / gradient_size
        previous_position = current_position
        if linesearch_type:
            learning_rate = linesearch(
                position=current_position,
                func=func,
                direction=direction_at_current_position,
                LB=LB,
                UB=UB,
            )
        delta_position = learning_rate * direction_at_current_position * gradient_size
        current_position = previous_position + delta_position
        iteration += 1
        condition_iteration = iteration > n_iterations
        condition_step = np.linalg.norm(delta_position) <= min_step_size
        condition_gradient = np.linalg.norm(gradient_size) <= min_grad_size
        condition = condition_iteration or condition_step or condition_gradient
    return position_best_so_far, func_value_best_so_far