import itertools

import sympy

from confusion_matrix import NumericalConfusionMatrix, SymbolicConfusionMatrix



def get_extrema_solution_vertices(exact_obj_tuple, approx_obj_tuple):

    symb_cm = SymbolicConfusionMatrix()

    num_exact = len(exact_obj_tuple)
    num_approx = len(approx_obj_tuple)

    # -------------------------------------------------------------------------
    # Check symbolic matrix
    symbolic_lhs_list = [None] * 4
    symbolic_rhs_list = [None] * 4

    for idx, m_obj in enumerate((*exact_obj_tuple, *approx_obj_tuple)):
        symbolic_lhs, symbolic_rhs = m_obj.get_symbolic_eqn(symb_cm)
        symbolic_lhs_list[idx] = symbolic_lhs
        symbolic_rhs_list[idx] = symbolic_rhs

    all_symbolic_lhs_matrix = sympy.Matrix(symbolic_lhs_list)
    all_symbolic_lhs_rank = all_symbolic_lhs_matrix.rank()
    if all_symbolic_lhs_rank < 4:
        return ("Symbolic rank < 4", None)
    # -------------------------------------------------------------------------

    numeric_lhs_list = [None] * 4
    numeric_rhs_list = [None] * 4

    # Populate the numeric matrix with the exact metric values
    for idx, m_obj in enumerate(exact_obj_tuple):
        numeric_lhs, numeric_rhs = m_obj.get_numeric_eqn()
        numeric_lhs_list[idx] = numeric_lhs
        numeric_rhs_list[idx] = numeric_rhs

    # Create a list of uncertainty tuples for all the approx metrics
    list_of_unc_tuples = [m_obj.uncertainty_range for m_obj in approx_obj_tuple]

    solution_vertex_dict = {}

    # Loop over each combination of uncertainty values ...
    for curr_approx_value_tuple, tuple_index in zip(itertools.product(*list_of_unc_tuples), itertools.product(range(2), repeat=num_approx)):

        # Add them to the numeric matrix
        for m_idx, v in enumerate(curr_approx_value_tuple):
            m_obj = approx_obj_tuple[m_idx]

            numeric_lhs, numeric_rhs = m_obj.get_eqn(v)
            numeric_lhs_list[m_idx+num_exact] = numeric_lhs
            numeric_rhs_list[m_idx+num_exact] = numeric_rhs

        # Check for rank
        all_numeric_lhs_matrix = sympy.Matrix(numeric_lhs_list)
        all_numeric_lhs_rank = all_numeric_lhs_matrix.rank()
        if all_numeric_lhs_rank < 4:
            return ("Numeric rank < 4", None)

        # Solve the linear system
        all_numeric_rhs_matrix = sympy.Matrix(numeric_rhs_list)
        curr_solution = sympy.linsolve((all_numeric_lhs_matrix, all_numeric_rhs_matrix))
        curr_solution_vertex = sympy.Matrix(list(curr_solution)[0])

        # Add the solution vertex to the solution dict
        solution_vertex_dict[tuple_index] = curr_solution_vertex

    return (None, solution_vertex_dict)


def get_intersection_of_AP(start_1, step_1,
                           start_2, step_2):

    # If one AP is "None", the resultant AP is just the other AP
    if step_1 is None:
        result_start = start_2
        result_step = step_2

    elif step_2 is None:
        result_start = start_1
        result_step = step_1

    else:

        # There will exist an intersection if and only if
        # the difference between the starts is an integral multiple of the GCD of the steps
        step_gcd = sympy.gcd(step_1, step_2)
        if not isinstance((start_1 - start_2) / step_gcd, sympy.Integer):
            return (None, None)

        # The resulting AP will have a step equal to the LCM of the two steps
        result_step = sympy.lcm(step_1, step_2)

        # Find the starting point of the new AP
        if start_1 > start_2:
            higher_start = start_1
            higher_step = step_1
            lower_start = start_2
            lower_step = step_2
        else:
            higher_start = start_2
            higher_step = step_2
            lower_start = start_1
            lower_step = step_1

        max_search_steps = result_step / higher_step

        for n in sympy.Range(max_search_steps):
            higher_val = higher_start + (n * higher_step)
            mult = (higher_val - lower_start) / lower_step
            if isinstance(mult, sympy.Integer):
                result_start = higher_val
                break

    return (result_start, result_step)


def exact_3_approx_1(exact_obj_tuple, approx_obj_tuple):

    esv_error_msg, extrema_solution_vertices = get_extrema_solution_vertices(exact_obj_tuple, approx_obj_tuple)

    if esv_error_msg is not None:
        return esv_error_msg, None

    # There are only two extrema solution vertices
    start_vertex = extrema_solution_vertices[(0,)]
    end_vertex = extrema_solution_vertices[(1,)]

    # The basis vector is just the vector from start to end
    basis_vector = end_vertex - start_vertex

    # If the basis_vector is all zeros, there is only one solution: The start vertex!
    if all([(bve == 0) for bve in basis_vector]):
        cm = NumericalConfusionMatrix(tp=start_vertex[0], fn=start_vertex[1], fp=start_vertex[2], tn=start_vertex[3])
        return(None, [cm])


    # Compute the p0 and step vectors for the arithmetic progressions (AP's) of the free param
    # Note: The free param step is designed such that p0 is the smallest positive number, and the step is always positive!
    p0_vec = [None] * 4
    step_vec = [None] * 4

    for idx in range(4):
        if basis_vector[idx] != 0:
            if basis_vector[idx] > 0:
                p0_vec[idx] = (start_vertex[idx].ceiling() - start_vertex[idx]) / basis_vector[idx]
                step_vec[idx] = 1 / basis_vector[idx]
            else:
                p0_vec[idx] = (start_vertex[idx].floor() - start_vertex[idx]) / basis_vector[idx]
                step_vec[idx] = -1 / basis_vector[idx]

    # If any of the p0 elements are > 1, this is an invalid solution
    for p0_elem in p0_vec:
        if p0_elem > 1:
            return("No valid CM in the given range", None)

    # Find the intersection of the 4 APs, two at a time
    temp_intersection_p0_vec = [None] * 2
    temp_intersection_step_vec = [None] * 2

    for set_idx in range(2):
        m_idx_1 = set_idx * 2
        m_idx_2 = m_idx_1 + 1

        temp_p0, temp_step = get_intersection_of_AP(p0_vec[m_idx_1], step_vec[m_idx_1],
                                                    p0_vec[m_idx_2], step_vec[m_idx_2])

        temp_intersection_p0_vec[set_idx] = temp_p0
        temp_intersection_step_vec[set_idx] = temp_step

    # Find the final intersection
    final_p0, final_step = get_intersection_of_AP(temp_intersection_p0_vec[0], temp_intersection_step_vec[0],
                                                  temp_intersection_p0_vec[1], temp_intersection_step_vec[1])

    # Final checks
    if final_p0 is None:
        return("No intersection exists for the APs", None)

    if final_p0 > 1:
        return("No valid CM in the given range", None)

    # Find the maximum value of the mult that allows `p` to be within the [0, 1] range
    max_mult = ((1 - final_p0) / final_step).floor()

    # Enumerate the resultant CMs
    cm_list = []
    for mult in sympy.Range(max_mult+1):

        curr_p = final_p0 + (mult * final_step)
        curr_cm_vector = start_vertex + (curr_p * basis_vector)

        curr_cm = NumericalConfusionMatrix(tp=int(curr_cm_vector[0]),
                                           fn=int(curr_cm_vector[1]),
                                           fp=int(curr_cm_vector[2]),
                                           tn=int(curr_cm_vector[3]))

        cm_list.append(curr_cm)

    return (None, cm_list)
