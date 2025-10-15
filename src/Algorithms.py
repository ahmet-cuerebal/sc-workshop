import sys
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import Problem_setting
import time
import random


class Solutions:
    def __init__(self, problem, allocated_work_instructions=None, name="Main"):
        self.problem = problem
        if allocated_work_instructions is None:
            allocated_work_instructions = []
        self.allocated_work_instructions = allocated_work_instructions
        self.objective_function = sys.maxsize
        self.best_found_objective_function = sys.maxsize
        self.name = name
        self.solution = None

    def copy_allocated_work_instructions(self):
        new_list = []
        for inner_list in self.allocated_work_instructions:
            new_inner_list = []
            for object in inner_list:
                new_inner_list.append(object)
            new_list.append(new_inner_list)
        return new_list

    def print_allocations(self):
        print("=" * 80)
        num_of_sc = self.problem.sc_number
        max_sc_name_length = max(len(str(sc)) for sc in self.problem.scs)

        for i, wi_list in enumerate(self.allocated_work_instructions):
            sc_name = str(self.problem.scs[i])

            sc_name_formatted = sc_name.ljust(max_sc_name_length)

            if wi_list is not None and len(wi_list) > 0:
                wi_str = ' '.join(str(wi) for wi in wi_list)
            else:
                wi_str = "No work instructions assigned"

            print(f"{sc_name_formatted} [ {wi_str} ]")
        print("=" * 80)
        print()

    def set_manual_assignment(self, assignment_text):
        """
        Set a manual assignment and calculate its objective function.

        Args:
            assignment_text: String in the format:
                SC0 [ WI12 WI7 WI11 ]
                SC1 [ WI10 WI4 WI13 ]
                SC2 [ WI3 WI8 WI9 ]
                ...

        Returns:
            tuple: (is_feasible, objective_value)
        """

        wi_map = {wi.name: wi for wi in self.problem.wis}

        self.allocated_work_instructions = [[] for _ in range(self.problem.sc_number)]

        lines = assignment_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split('[')
            if len(parts) != 2:
                print(f"Warning: Skipping invalid line: {line}")
                continue

            sc_part = parts[0].strip()
            if not sc_part.startswith('SC'):
                print(f"Warning: Invalid SC identifier: {sc_part}")
                continue

            try:
                sc_index = int(sc_part[2:])
            except ValueError:
                print(f"Warning: Invalid SC index: {sc_part}")
                continue

            if sc_index >= self.problem.sc_number:
                print(f"Warning: SC index {sc_index} out of range (max: {self.problem.sc_number - 1})")
                continue

            wi_part = parts[1].strip().rstrip(']').strip()

            if not wi_part:
                continue

            wi_names = wi_part.split()

            for wi_name in wi_names:
                if wi_name not in wi_map:
                    print(f"Warning: Unknown WI: {wi_name}")
                    continue

                wi_obj = wi_map[wi_name]
                self.allocated_work_instructions[sc_index].append(wi_obj)

        assigned_wis = set()
        for route in self.allocated_work_instructions:
            for wi in route:
                if wi.name in assigned_wis:
                    print(f"ERROR: {wi.name} is assigned multiple times!")
                    return False, None
                assigned_wis.add(wi.name)

        all_wi_names = {wi.name for wi in self.problem.wis}
        unassigned = all_wi_names - assigned_wis

        if unassigned:
            print(f"ERROR: The following WIs are not assigned: {sorted(unassigned)}")
            return False, None

        is_feasible = self.calculation_of_obj()

        if is_feasible:

            return True, self.objective_function
        else:
            print(f"✗ Solution is INFEASIBLE (time window violations)")
            return False, None

    def calculation_of_obj(self, to_beat=None):
        """
        Calculate the objective function for the current solution.

        Returns:
            True if solution is feasible, False if time window violations occur
        """

        objective = 0

        # Reset SCs to initial state
        for sc in self.problem.scs:
            sc.last_WI_finish_time = sc.start_time
            sc.last_WI = None
            sc.next_WI = None
            sc.last_position = sc.start_position

        # Reset WIs
        for wi in self.problem.wis:
            wi.planned_start_time = None
            wi.planned_end_time = None
            wi.planned_time_is_calculated = False
            wi.sc = None

        # Define next WI for each SC at the beginning
        for i, sc in enumerate(self.problem.scs):
            if not self.allocated_work_instructions[i]:
                sc.next_WI = None
            else:
                sc.next_WI = self.allocated_work_instructions[i][0]

        # Assign indices to WIs (sc_index, position_in_route)
        for sc_index, sc_route in enumerate(self.allocated_work_instructions):
            for position, wi in enumerate(sc_route):
                wi.assignment_index = (sc_index, position)

        # Process all WIs in sequence
        remaining_wi = True

        while remaining_wi:
            remaining_wi = False

            for current_SC in self.problem.scs:
                wi = current_SC.next_WI

                if wi is None:
                    continue

                remaining_wi = True

                # Determine previous location
                if current_SC.last_WI is None:
                    previous_location = current_SC  # SC at starting position
                else:
                    previous_location = current_SC.last_WI  # SC at last WI's final location

                # Calculate empty travel time (from previous location to WI's origin)
                empty_travel_time = self.problem.durations[previous_location][wi]

                # When can we arrive at WI's origin?
                previous_end_time = current_SC.last_WI_finish_time
                arrival_at_origin = previous_end_time + empty_travel_time

                # Planned pickup time (must respect earliest start time)
                planned_pickup_time = max(arrival_at_origin, wi.earliest_move_start_time)
                wi.planned_start_time = planned_pickup_time

                # Calculate dropoff time (pickup + operation time)
                arrival_at_destination = planned_pickup_time + wi.operation_time

                # Planned dropoff time (must respect earliest end time - waiting if needed)
                planned_dropoff_time = max(arrival_at_destination, wi.earliest_move_end_time)
                wi.planned_end_time = planned_dropoff_time

                # Check pickup window constraint
                if wi.planned_start_time > wi.latest_move_start_time:
                    # Violation: cannot pick up within pickup window
                    return False

                # Check dropoff window constraint
                if wi.planned_end_time > wi.latest_move_end_time:
                    # Violation: cannot dropoff within dropoff window
                    return False

                objective += wi.planned_end_time

                # Early termination if objective exceeds the best one so far
                if to_beat is not None and objective > to_beat:
                    return False

                # Update WI information
                wi.sc = current_SC
                wi.planned_time_is_calculated = True

                # Update SC state
                current_SC.last_position = wi.final_location
                current_SC.last_WI_finish_time = wi.planned_end_time
                current_SC.last_WI = wi

                # Set next WI for this SC
                sc_index, position = wi.assignment_index
                if wi != self.allocated_work_instructions[sc_index][-1]:
                    # Not the last WI in route
                    current_SC.next_WI = self.allocated_work_instructions[sc_index][position + 1]
                else:
                    # Last WI in route
                    current_SC.next_WI = None

        # Store objective value
        self.objective_function = objective

        return True

    def the_greedy_V2(self):
        """
        Greedy algorithm that assigns WIs to SCs.
        Returns True if feasible, False if any WI cannot be assigned.
        """
        num_of_SCs = self.problem.sc_number
        num_of_WIs = self.problem.wis_number

        # Reset SCs to initial state
        for sc in self.problem.scs:
            sc.last_WI_finish_time = sc.start_time
            sc.last_WI = None
            sc.next_WI = None
            sc.current_position = sc.start_position
            sc.last_position = sc

        # Reset WIs
        for wi in self.problem.wis:
            wi.planned_start_time = None
            wi.planned_end_time = None
            wi.sc = None
            wi.assigned = False

        # Initialize allocation lists
        self.allocated_work_instructions = [[] for _ in range(num_of_SCs)]

        # Track remaining unassigned WIs
        remaining_wis = set(self.problem.wis)
        objective = 0

        # Assign WIs one by one, picking best (SC, WI) pair each iteration
        while remaining_wis:
            best_sc = None
            best_wi = None
            best_dropoff_time = float('inf')
            best_pickup_time = None
            best_key = None

            # Evaluate all (SC, WI) pairs and pick the best one
            for sc in self.problem.scs:
                # Get SC's current location
                if sc.last_WI is None:
                    current_location = sc
                else:
                    current_location = sc.last_WI

                current_time = sc.last_WI_finish_time

                for wi in remaining_wis:
                    # Calculate empty travel time to WI's origin
                    travel_to_origin = self.problem.durations[current_location][wi]
                    arrival_at_origin = current_time + travel_to_origin

                    # Pickup time respects earliest start time
                    actual_pickup_time = max(arrival_at_origin, wi.earliest_move_start_time)

                    # Check pickup window feasibility
                    if actual_pickup_time > wi.latest_move_start_time:
                        continue

                    # Calculate arrival at destination
                    arrival_at_destination = actual_pickup_time + wi.operation_time

                    # Dropoff time for waiting at dropoff window
                    actual_dropoff_time = max(arrival_at_destination, wi.earliest_move_end_time)

                    # Check dropoff window feasibility
                    if actual_dropoff_time > wi.latest_move_end_time:
                        continue

                    # Selection criteria: earliest dropoff, then earliest pickup, then shortest empty travel
                    selection_key = (actual_dropoff_time, actual_pickup_time, travel_to_origin)

                    if best_key is None or selection_key < best_key:
                        best_sc = sc
                        best_wi = wi
                        best_pickup_time = actual_pickup_time
                        best_dropoff_time = actual_dropoff_time
                        best_key = selection_key

            # Check if we found a feasible assignment
            if best_sc is None:
                # No feasible SC found for any remaining WI - return False
                return False

            # Assign best WI to best SC
            best_wi.planned_start_time = best_pickup_time
            best_wi.planned_end_time = best_dropoff_time
            best_wi.sc = best_sc
            best_wi.assigned = True

            # Update SC state
            best_sc.last_WI = best_wi
            best_sc.last_WI_finish_time = best_wi.planned_end_time
            best_sc.last_position = best_wi

            # Add to allocation list
            self.allocated_work_instructions[best_sc.index].append(best_wi)

            objective += best_wi.planned_end_time

            # Remove from remaining
            remaining_wis.remove(best_wi)

        # All WIs assigned successfully
        self.objective_function = objective

        return True

    def the_greedy(self):
        """
        Greedy algorithm that assigns WIs to SCs.
        Strategy: Process WIs by urgency (earliest start time), find best SC for each.
        Returns True if feasible, False if any WI cannot be assigned.
        """
        num_of_SCs = self.problem.sc_number
        num_of_WIs = self.problem.wis_number

        # Reset SCs to initial state
        for sc in self.problem.scs:
            sc.last_WI_finish_time = sc.start_time
            sc.last_WI = None
            sc.next_WI = None
            sc.current_position = sc.start_position
            sc.last_position = sc

        # Reset WIs
        for wi in self.problem.wis:
            wi.planned_start_time = None
            wi.planned_end_time = None
            wi.sc = None
            wi.assigned = False

        # Initialize allocation lists
        self.allocated_work_instructions = [[] for _ in range(num_of_SCs)]

        # Sort WIs by urgency: latest start time
        sorted_wis = sorted(self.problem.wis,
                            key=lambda x: (x.latest_move_start_time))

        objective = 0

        # Process each WI in order of urgency
        for wi in sorted_wis:
            best_sc = None
            best_dropoff_time = float('inf')
            best_pickup_time = None
            best_key = None

            # Find best SC for this specific WI
            for sc in self.problem.scs:
                # Get SC's current location
                if sc.last_WI is None:
                    current_location = sc
                else:
                    current_location = sc.last_WI

                current_time = sc.last_WI_finish_time

                # Calculate empty travel time to WI's origin
                travel_to_origin = self.problem.durations[current_location][wi]
                arrival_at_origin = current_time + travel_to_origin

                # Pickup time respects earliest start time
                actual_pickup_time = max(arrival_at_origin, wi.earliest_move_start_time)

                # Check pickup window feasibility
                if actual_pickup_time > wi.latest_move_start_time:
                    continue

                # Calculate arrival at destination
                arrival_at_destination = actual_pickup_time + wi.operation_time

                # Dropoff time for waiting at dropoff window
                actual_dropoff_time = max(arrival_at_destination, wi.earliest_move_end_time)

                # Check dropoff window feasibility
                if actual_dropoff_time > wi.latest_move_end_time:
                    continue

                # Selection criteria: earliest dropoff, then earliest pickup, then shortest empty travel
                selection_key = (actual_dropoff_time, actual_pickup_time, travel_to_origin)

                if best_key is None or selection_key < best_key:
                    best_sc = sc
                    best_pickup_time = actual_pickup_time
                    best_dropoff_time = actual_dropoff_time
                    best_key = selection_key

            # Check if we found a feasible SC for this WI
            if best_sc is None:
                # No feasible SC found for this WI - return False
                return False

            # Assign WI to best SC
            wi.planned_start_time = best_pickup_time
            wi.planned_end_time = best_dropoff_time
            wi.sc = best_sc
            wi.assigned = True

            # Update SC state
            best_sc.last_WI = wi
            best_sc.last_WI_finish_time = wi.planned_end_time
            best_sc.last_position = wi

            # Add to allocation list
            self.allocated_work_instructions[best_sc.index].append(wi)

            # Accumulate objective
            objective += wi.planned_end_time

        # All WIs assigned successfully
        self.objective_function = objective

        return True

    def the_greedy_randomized(self, alpha=0.3, seed=None):
        """
        Randomized greedy for GRASP construction phase.
        Strategy: Process WIs by urgency, randomly select SC from RCL of best candidates.
        Accepts infeasible assignments with penalty.

        Args:
            alpha: RCL size as fraction (0.3 = top 30% of SCs)
            seed: Random seed for reproducibility

        Returns:
            objective: Objective value (may include penalties)
        """
        import random
        import sys

        if seed is not None:
            random.seed(seed)

        num_of_SCs = self.problem.sc_number
        num_of_WIs = self.problem.wis_number

        # Reset SCs to initial state
        for sc in self.problem.scs:
            sc.last_WI_finish_time = sc.start_time
            sc.last_WI = None
            sc.next_WI = None
            sc.current_position = sc.start_position
            sc.last_position = sc

        # Reset WIs
        for wi in self.problem.wis:
            wi.planned_start_time = None
            wi.planned_end_time = None
            wi.sc = None
            wi.assigned = False

        # Initialize allocation lists
        self.allocated_work_instructions = [[] for _ in range(num_of_SCs)]

        # Sort WIs by urgency: latest start time
        sorted_wis = sorted(self.problem.wis, key=lambda x: x.latest_move_start_time)

        objective = 0
        infeasible_count = 0

        # Process each WI in order of urgency
        for wi in sorted_wis:
            # Build candidate list of all SCs for this WI
            candidates = []

            for sc in self.problem.scs:
                # Get SC's current location
                if sc.last_WI is None:
                    current_location = sc
                else:
                    current_location = sc.last_WI

                current_time = sc.last_WI_finish_time

                # Calculate empty travel time to WI's origin
                travel_to_origin = self.problem.durations[current_location][wi]
                arrival_at_origin = current_time + travel_to_origin

                # Pickup time respects earliest start time
                actual_pickup_time = max(arrival_at_origin, wi.earliest_move_start_time)

                # Check pickup window feasibility
                pickup_feasible = actual_pickup_time <= wi.latest_move_start_time

                # Calculate arrival at destination
                arrival_at_destination = actual_pickup_time + wi.operation_time

                # Dropoff time accounts for waiting at dropoff window
                actual_dropoff_time = max(arrival_at_destination, wi.earliest_move_end_time)

                # Check dropoff window feasibility
                dropoff_feasible = actual_dropoff_time <= wi.latest_move_end_time

                # Selection criteria
                selection_key = (actual_dropoff_time, actual_pickup_time, travel_to_origin)

                # Add to candidates (both feasible and infeasible)
                candidates.append({
                    'sc': sc,
                    'pickup_time': actual_pickup_time,
                    'dropoff_time': actual_dropoff_time,
                    'selection_key': selection_key,
                    'is_feasible': pickup_feasible and dropoff_feasible
                })

            # Separate feasible and infeasible candidates
            feasible_candidates = [c for c in candidates if c['is_feasible']]

            if feasible_candidates:
                # Use feasible candidates
                candidate_list = feasible_candidates
            else:
                # No feasible candidates - use all and mark as infeasible
                candidate_list = candidates
                infeasible_count += 1

            # Sort by selection key (best first)
            candidate_list.sort(key=lambda x: x['selection_key'])

            # Build Restricted Candidate List (RCL) - cardinality-based
            rcl_size = max(1, int(len(candidate_list) * alpha))
            rcl = candidate_list[:rcl_size]

            # Randomly select from RCL
            selected = random.choice(rcl)

            # Extract selection
            selected_sc = selected['sc']
            selected_pickup_time = selected['pickup_time']
            selected_dropoff_time = selected['dropoff_time']
            is_feasible = selected['is_feasible']

            # Assign WI to selected SC
            wi.planned_start_time = selected_pickup_time
            wi.planned_end_time = selected_dropoff_time
            wi.sc = selected_sc
            wi.assigned = True

            # Update SC state
            selected_sc.last_WI = wi
            selected_sc.last_WI_finish_time = wi.planned_end_time
            selected_sc.last_position = wi

            # Add to allocation list
            self.allocated_work_instructions[selected_sc.index].append(wi)

            if is_feasible:
                objective += wi.planned_end_time
            else:
                # Penalty for infeasible assignment
                objective += sys.maxsize / num_of_WIs

        self.objective_function = objective

        # if infeasible_count > 0:
        #     print(f"  Randomized construction: {infeasible_count}/{num_of_WIs} infeasible")

        return objective

    def the_greedy_randomized_V2(self, alpha=0.3, seed=None):
        """
        Randomized greedy algorithm for GRASP construction phase.
        Uses cardinality-based RCL to select from top alpha% of candidates.
        Accepts infeasible assignments with penalty.

        Args:
            alpha: RCL size as fraction (0.3 = top 30% of candidates)
            seed: Random seed for reproducibility

        Returns:
            objective: Objective value (may include penalties for infeasible assignments)
        """
        import random
        import sys

        if seed is not None:
            random.seed(seed)

        num_of_SCs = self.problem.sc_number
        num_of_WIs = self.problem.wis_number

        # Reset SCs to initial state
        for sc in self.problem.scs:
            sc.last_WI_finish_time = sc.start_time
            sc.last_WI = None
            sc.next_WI = None
            sc.current_position = sc.start_position
            sc.last_position = sc

        # Reset WIs
        for wi in self.problem.wis:
            wi.planned_start_time = None
            wi.planned_end_time = None
            wi.sc = None
            wi.assigned = False

        # Initialize allocation lists
        self.allocated_work_instructions = [[] for _ in range(num_of_SCs)]

        # Track remaining unassigned WIs
        remaining_wis = set(self.problem.wis)
        objective = 0
        infeasible_count = 0

        # Assign WIs one by one
        while remaining_wis:
            # Build candidate list: all feasible (SC, WI) pairs
            candidates = []

            for sc in self.problem.scs:
                # Get SC's current location
                if sc.last_WI is None:
                    current_location = sc
                else:
                    current_location = sc.last_WI

                current_time = sc.last_WI_finish_time

                for wi in remaining_wis:
                    # Calculate empty travel time to WI's origin
                    travel_to_origin = self.problem.durations[current_location][wi]
                    arrival_at_origin = current_time + travel_to_origin

                    # Pickup time respects earliest start time
                    actual_pickup_time = max(arrival_at_origin, wi.earliest_move_start_time)

                    # Check pickup window feasibility
                    pickup_feasible = actual_pickup_time <= wi.latest_move_start_time

                    # Calculate arrival at destination
                    arrival_at_destination = actual_pickup_time + wi.operation_time

                    # Dropoff time accounts for waiting at dropoff window
                    actual_dropoff_time = max(arrival_at_destination, wi.earliest_move_end_time)

                    # Check dropoff window feasibility
                    dropoff_feasible = actual_dropoff_time <= wi.latest_move_end_time

                    # Selection criteria: earliest dropoff, then earliest pickup, then shortest empty travel
                    selection_key = (actual_dropoff_time, actual_pickup_time, travel_to_origin)

                    # Add to candidates (both feasible and infeasible)
                    candidates.append({
                        'sc': sc,
                        'wi': wi,
                        'pickup_time': actual_pickup_time,
                        'dropoff_time': actual_dropoff_time,
                        'selection_key': selection_key,
                        'is_feasible': pickup_feasible and dropoff_feasible
                    })

            # Separate feasible and infeasible candidates
            feasible_candidates = [c for c in candidates if c['is_feasible']]

            if feasible_candidates:
                # Use feasible candidates for RCL
                candidate_list = feasible_candidates
            else:
                # No feasible candidates - use all candidates and mark as infeasible
                candidate_list = candidates
                infeasible_count += 1

            # Sort by selection key (best first)
            candidate_list.sort(key=lambda x: x['selection_key'])

            # Build Restricted Candidate List (RCL) - cardinality-based
            rcl_size = max(1, int(len(candidate_list) * alpha))
            rcl = candidate_list[:rcl_size]

            # Randomly select from RCL
            selected = random.choice(rcl)

            # Extract selected assignment
            selected_sc = selected['sc']
            selected_wi = selected['wi']
            selected_pickup_time = selected['pickup_time']
            selected_dropoff_time = selected['dropoff_time']
            is_feasible = selected['is_feasible']

            # Assign WI to selected SC
            selected_wi.planned_start_time = selected_pickup_time
            selected_wi.planned_end_time = selected_dropoff_time
            selected_wi.sc = selected_sc
            selected_wi.assigned = True

            # Update SC state
            selected_sc.last_WI = selected_wi
            selected_sc.last_WI_finish_time = selected_wi.planned_end_time
            selected_sc.last_position = selected_wi

            # Add to allocation list
            self.allocated_work_instructions[selected_sc.index].append(selected_wi)

            # Accumulate objective
            if is_feasible:
                objective += selected_wi.planned_end_time
            else:
                # Penalty for infeasible assignment
                objective += sys.maxsize / num_of_WIs  # Scaled penalty

            # Remove from remaining
            remaining_wis.remove(selected_wi)

        # Store objective
        self.objective_function = objective

        # if infeasible_count > 0:
        #     print(f"  Randomized construction: {infeasible_count}/{num_of_WIs} infeasible assignments")

        return objective

    def local_search(self, time_limit=180, time_at_start=None):
        if time_at_start is None:
            time_at_start = time.time()
        number_of_SCs = self.problem.sc_number
        has_improvement = False

        for i in range(0, number_of_SCs):
            for j in range(0, number_of_SCs):
                if i != j:
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement
                    was_improvement = self.LS_2exchange_best_found(i, j, time_limit, time_at_start)
                    if was_improvement:
                        has_improvement = True
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement

        for i in range(0, number_of_SCs):
            for j in range(0, number_of_SCs):
                if i != j:
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement
                    was_improvement = self.LS_2relocate_best_found(i, j, time_limit, time_at_start)
                    if was_improvement:
                        has_improvement = True
                    current_time = time.time() - time_at_start
                    if current_time > time_limit:
                        return has_improvement

        return has_improvement

    def LS_2exchange_best_found(self, sc_index1, sc_index2, time_limit, time_at_start):
        has_improvement = False
        current_objective_function = self.objective_function
        number_of_WIs1 = len(self.allocated_work_instructions[sc_index1])
        number_of_WIs2 = len(self.allocated_work_instructions[sc_index2])
        best_solution = Solutions(self.problem, name="Best_one")
        best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
        best_solution.objective_function = current_objective_function
        best_objective_function = current_objective_function

        for index1 in range(number_of_WIs1):
            current_time = time.time() - time_at_start
            if current_time > time_limit:
                break  # Exit the loop if time limit is exceeded
            for index2 in range(number_of_WIs2):
                current_time = time.time() - time_at_start
                if current_time > time_limit:
                    break  # Exit the loop if time limit is exceeded
                # Try swapping WIs index1 and index2
                new_solution = Solutions(self.problem, name="New_one")
                new_solution.allocated_work_instructions = self.copy_allocated_work_instructions()

                new_solution.allocated_work_instructions[sc_index1][index1], \
                    new_solution.allocated_work_instructions[sc_index2][index2] = \
                    new_solution.allocated_work_instructions[sc_index2][index2], \
                        new_solution.allocated_work_instructions[sc_index1][index1]

                improvement = new_solution.calculation_of_obj(best_objective_function)

                if improvement:
                    best_solution.allocated_work_instructions = new_solution.copy_allocated_work_instructions()
                    best_solution.objective_function = new_solution.objective_function
                    best_objective_function = new_solution.objective_function
                    has_improvement = True

        if has_improvement:
            self.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
            self.objective_function = best_objective_function
            return True

        # If no improvement was found, return False
        return False

    def LS_2relocate_best_found(self, sc_index1, sc_index2, time_limit, time_at_start):
        has_improvement = False
        current_objective_function = self.objective_function
        number_of_WIs1 = len(self.allocated_work_instructions[sc_index1])
        if number_of_WIs1 == 1:
            return False
        number_of_WIs2 = len(self.allocated_work_instructions[sc_index2])
        best_solution = Solutions(self.problem, name="Best_one")
        best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
        best_solution.objective_function = current_objective_function
        best_objective_function = current_objective_function

        for index1 in range(number_of_WIs1):
            current_time = time.time() - time_at_start
            if current_time > time_limit:
                break  # Exit the loop if time limit is exceeded
            for index2 in range(number_of_WIs2):
                current_time = time.time() - time_at_start
                if current_time > time_limit:
                    break  # Exit the loop if time limit is exceeded
                # Try relocating WI index1 from sc_index1 to sc_index2
                new_solution = Solutions(self.problem, name="New_one")
                new_solution.allocated_work_instructions = self.copy_allocated_work_instructions()

                WI_to_relocate = new_solution.allocated_work_instructions[sc_index1][index1]
                new_solution.allocated_work_instructions[sc_index1].remove(WI_to_relocate)
                new_solution.allocated_work_instructions[sc_index2].insert(index2, WI_to_relocate)

                improvement = new_solution.calculation_of_obj(best_objective_function)

                if improvement:
                    best_solution.allocated_work_instructions = new_solution.copy_allocated_work_instructions()
                    best_solution.objective_function = new_solution.objective_function
                    best_objective_function = new_solution.objective_function
                    has_improvement = True

        # If there's an improvement, update the allocated WIs and return True
        if has_improvement:
            self.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
            self.objective_function = best_objective_function
            return True

        # If no improvement was found, return False
        return False

    def LS_2opt_2route_best_found(self, sc_index1, sc_index2, time_limit, time_at_start):
        has_improvement = False
        current_objective_function = self.objective_function
        number_of_WIs1 = len(self.allocated_work_instructions[sc_index1])
        number_of_WIs2 = len(self.allocated_work_instructions[sc_index2])
        best_solution = Solutions(self.problem, name="Best_one")
        best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
        best_solution.objective_function = current_objective_function
        best_objective_function = current_objective_function

        for index1 in range(1, number_of_WIs1):
            current_time = time.time() - time_at_start
            if current_time > time_limit:
                break  # Exit the loop if time limit is exceeded
            for index2 in range(1, number_of_WIs2):
                current_time = time.time() - time_at_start
                if current_time > time_limit:
                    break  # Exit the loop if time limit is exceeded
                # Create a new solution by swapping edges between two routes
                new_solution = Solutions(self.problem, name="New_one")
                new_solution.allocated_work_instructions = self.copy_allocated_work_instructions()

                # Swap WIs index1 of sc_index1 with index2 of sc_index2
                temp = new_solution.allocated_work_instructions[sc_index1][index1:]
                new_solution.allocated_work_instructions[sc_index1][index1:] = new_solution.allocated_work_instructions[
                                                                                   sc_index2][index2:]
                new_solution.allocated_work_instructions[sc_index2][index2:] = temp

                improvement = new_solution.calculation_of_obj(best_objective_function)

                if improvement:
                    best_solution.allocated_work_instructions = new_solution.copy_allocated_work_instructions()
                    best_solution.objective_function = new_solution.objective_function
                    best_objective_function = new_solution.objective_function
                    has_improvement = True

            # If there's an improvement, update the allocated WIs and return True
        if has_improvement:
            self.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
            self.objective_function = best_objective_function
            return True

        # If no improvement was found, return False
        return False

    def shake_functions(self, m):

        if m == 1:
            is_shaken = self.shake_random_2_relocation()
        elif m == 2:
            is_shaken = self.shake_random_2_exchange()
        else:
            is_shaken = self.shake_random_3_relocation()

        return is_shaken

    def shake_random_3_relocation(self):
        number_of_SCs = self.problem.sc_number
        if number_of_SCs < 3:
            return False  # Cannot perform 3-opt

        # Randomly select three different SCs
        sc_index1, sc_index2, sc_index3 = random.sample(range(number_of_SCs), 3)

        # Get the number of jobs for each selected SC
        number_of_jobs1 = len(self.allocated_work_instructions[sc_index1])
        number_of_jobs2 = len(self.allocated_work_instructions[sc_index2])
        number_of_jobs3 = len(self.allocated_work_instructions[sc_index3])

        if number_of_jobs1 == 0 or number_of_jobs2 == 0 or number_of_jobs3 == 0:
            return False  # Cannot perform 3-opt with empty SCs

        # Randomly select one job from each SC
        job_index1 = random.randint(0, number_of_jobs1 - 1)
        job_index2 = random.randint(0, number_of_jobs2 - 1)
        job_index3 = random.randint(0, number_of_jobs3 - 1)

        # Remove the jobs from their original positions
        temp1 = self.allocated_work_instructions[sc_index1].pop(job_index1)
        temp2 = self.allocated_work_instructions[sc_index2].pop(job_index2)
        temp3 = self.allocated_work_instructions[sc_index3].pop(job_index3)

        # Place the removed jobs into new positions
        self.allocated_work_instructions[sc_index1].insert(job_index1, temp2)
        self.allocated_work_instructions[sc_index2].insert(job_index2, temp3)
        self.allocated_work_instructions[sc_index3].insert(job_index3, temp1)

        feasible = self.calculation_of_obj()

        if not feasible:
            return False

        return True

    def shake_random_2_relocation(self):
        number_of_SCs = self.problem.sc_number
        if number_of_SCs < 2:  # At least two SCs are needed for a relocation move
            return False

        # Randomly select a source SC
        sc_index_source = random.randint(0, number_of_SCs - 1)

        # the number of jobs for the source SC
        number_of_jobs_source = len(self.allocated_work_instructions[sc_index_source])

        if number_of_jobs_source <= 1:
            return False  # Cannot perform relocation, an empty source SC

        # randomly select a job from the source SC
        job_index_source = random.randint(0, number_of_jobs_source - 1)

        # Remove the selected job from the source SC and save it!
        temp_job = self.allocated_work_instructions[sc_index_source].pop(job_index_source)

        # Randomly select a destination SC
        sc_index_dest = random.choice([i for i in range(number_of_SCs) if i != sc_index_source])

        # the number of jobs for the destination SC
        number_of_jobs_dest = len(self.allocated_work_instructions[sc_index_dest])

        # Randomly select a position to insert the job in the destination SC
        job_index_dest = random.randint(0, number_of_jobs_dest)  # The position can be at the end?

        # relocate the job at the selected position in the destination SC
        self.allocated_work_instructions[sc_index_dest].insert(job_index_dest, temp_job)

        feasible = self.calculation_of_obj()

        if not feasible:
            return False

        return True

    def shake_random_2_exchange(self):
        number_of_SCs = self.problem.sc_number
        if number_of_SCs < 2:  # At least two SCs are needed for a 2-exchange move
            return False

        # Randomly select the first SC
        sc_index_1 = random.randint(0, number_of_SCs - 1)
        number_of_jobs_1 = len(self.allocated_work_instructions[sc_index_1])

        # check the number of job of SC1
        if number_of_jobs_1 == 0:
            return False

        # select a container
        job_index_1 = random.randint(0, number_of_jobs_1 - 1)

        # second sc
        sc_index_2 = random.choice([i for i in range(number_of_SCs) if i != sc_index_1])
        number_of_jobs_2 = len(self.allocated_work_instructions[sc_index_2])

        # check the job number of the second sc
        if number_of_jobs_2 == 0:
            return False

        # select a container from the second sc
        job_index_2 = random.randint(0, number_of_jobs_2 - 1)

        # exchange them
        self.allocated_work_instructions[sc_index_1][job_index_1], self.allocated_work_instructions[sc_index_2][job_index_2] = \
        self.allocated_work_instructions[sc_index_2][job_index_2], self.allocated_work_instructions[sc_index_1][job_index_1]

        feasible = self.calculation_of_obj()

        if not feasible:
            return False

        return True

    def GRASP(self, time_limit=60):

        grasp_start_time = time.time()

        # Initial solution with greedy
        feasible_greedy = self.the_greedy()
        print("greedy:", feasible_greedy)
        # Apply local search to initial solution
        self.local_search(time_limit=time_limit, time_at_start=grasp_start_time)
        ls_time = time.time() - grasp_start_time
        ls_obj_val = self.objective_function
        print("ls:", ls_obj_val)

        # Store best solution
        best_solution = Solutions(self.problem, name="Best Grasp")
        best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
        best_solution.objective_function = self.objective_function
        best_ever_grasp = best_solution.objective_function
        best_solution_found_time = ls_time

        # Alpha parameters
        alphas = np.linspace(0, 1, num=11)
        best_solution_found_time = ls_time

        for i in range(1, 1001):
            alpha = random.choice(alphas)

            greedy_randomized_value = self.the_greedy_randomized(alpha)
            self.objective_function = greedy_randomized_value
            self.local_search(time_limit=time_limit, time_at_start=grasp_start_time)
            value_to_comp = self.objective_function
            current_time = time.time() - grasp_start_time

            # Update best solution
            if value_to_comp < best_ever_grasp:
                best_ever_grasp = value_to_comp
                best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
                best_solution.objective_function = best_ever_grasp
                best_solution_found_time = current_time
                print(best_ever_grasp)

            if current_time > time_limit:
                break

        grasp_total_time = time.time() - grasp_start_time

        # Copy best solution back to self
        self.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
        self.objective_function = best_solution.objective_function
        self.best_found_objective_function = best_solution.objective_function

        return best_solution

    def VNS(self, m_max=3, time_limit=60, number_of_iterations=10000):
        """
        Variable Neighborhood Search (VNS) algorithm.

        Args:
            m_max: Maximum neighborhood size
            time_limit: Time limit in seconds
            number_of_iterations: Maximum number of iterations

        Returns:
        """
        vns_start_time = time.time()

        # Initial solution with greedy
        feasible_greedy = self.the_greedy()
        print("greedy:", feasible_greedy)
        # Apply local search to initial solution
        self.local_search(time_limit=time_limit, time_at_start=vns_start_time)
        ls_time = time.time() - vns_start_time
        ls_obj_val = self.objective_function
        print("ls:", ls_obj_val)

        # Store best solution
        best_solution = Solutions(self.problem, name="Best VNS")
        best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
        best_solution.objective_function = self.objective_function
        best_ever_vns = best_solution.objective_function
        best_solution_found_time = ls_time

        i = 0
        time_limit_reached = False

        while i < number_of_iterations:
            m = 1

            while m <= m_max:
                # Create new solution for shaking
                new_solution = Solutions(self.problem, name="Shaked Solution")
                new_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
                new_solution.objective_function = self.objective_function

                # Apply shake (neighborhood m)
                shake_success = new_solution.shake_functions(m)

                if not shake_success:
                    # Shake resulted in infeasible solution
                    self.allocated_work_instructions = new_solution.copy_allocated_work_instructions()
                    self.objective_function = sys.maxsize  # Penalty for infeasibility
                else:
                    # Shake was successful
                    self.allocated_work_instructions = new_solution.copy_allocated_work_instructions()
                    self.objective_function = new_solution.objective_function

                # Apply local search to shaken solution
                self.local_search(time_limit=time_limit, time_at_start=vns_start_time)
                new_obj = self.objective_function
                current_time = time.time() - vns_start_time

                # Check for improvement
                if new_obj < best_ever_vns:
                    best_ever_vns = new_obj
                    best_solution.allocated_work_instructions = self.copy_allocated_work_instructions()
                    best_solution.objective_function = best_ever_vns
                    best_solution_found_time = current_time
                    print(best_ever_vns)

                    m = 1  # Reset to first neighborhood
                else:
                    m += 1  # Try next neighborhood

                # Check time limit
                if current_time > time_limit:
                    time_limit_reached = True
                    break

            i += 1

        vns_total_time = time.time() - vns_start_time

        # Copy best solution back to self
        self.allocated_work_instructions = best_solution.copy_allocated_work_instructions()
        self.objective_function = best_solution.objective_function
        self.best_found_objective_function = best_solution.objective_function

        return best_solution

    def solve_with_gurobi(self, time_limit=60, verbose=False):
        """
        Solve the simple SC routing problem with hard time windows
        Objective: Minimize the total turnaround time of WIs

        Args:
            time_limit: Maximum solving time in seconds
            verbose: Whether to print output

        Returns:
            True if solution found, False otherwise
        """
        model = gp.Model("SimpleSCRouting")

        # if not verbose:
        #     model.Params.OutputFlag = 0

        # Extract data
        SCs = self.problem.scs
        WIs = self.problem.wis
        M = self.problem.horizon * 2  # Big M for constraints

        # Build arc set A (all possible movements)
        A = []
        d = {}  # Empty travel times

        # Arcs from SC initial positions to WI origins
        for sc in SCs:
            for wi in WIs:
                if sc in self.problem.durations and wi in self.problem.durations[sc]:
                    arc = (sc, wi)
                    A.append(arc)
                    d[arc] = self.problem.durations[sc][wi]

        # Arcs from WI destinations to other WI origins
        for wi_i in WIs:
            for wi_j in WIs:
                if wi_i != wi_j:
                    if wi_i in self.problem.durations and wi_j in self.problem.durations[wi_i]:
                        arc = (wi_i, wi_j)
                        A.append(arc)
                        d[arc] = self.problem.durations[wi_i][wi_j]

        # Decision variables
        # x[arc] = 1 if arc is used in solution
        x = {}
        for arc in A:
            x[arc] = model.addVar(vtype=GRB.BINARY, name=f"x_{arc[0]}_{arc[1]}")

        # Time variables
        t_s = {}  # Start time of each WI
        t_e = {}  # End time of each WI

        for wi in WIs:
            t_s[wi] = model.addVar(lb=0, name=f"t_s_{wi}")
            t_e[wi] = model.addVar(lb=0, name=f"t_e_{wi}")

        # OBJECTIVE: Minimize total end times
        model.setObjective(
            gp.quicksum(t_e[wi] for wi in WIs),
            GRB.MINIMIZE
        )

        # CONSTRAINTS

        # 1. Each WI must be assigned to exactly one predecessor
        for wi in WIs:
            incoming_arcs = [arc for arc in A if arc[1] == wi]
            model.addConstr(
                gp.quicksum(x[arc] for arc in incoming_arcs) == 1,
                name=f"assign_{wi}"
            )

        # 2. Each SC can have at most one immediate successor
        for sc in SCs:
            outgoing_arcs = [arc for arc in A if arc[0] == sc]
            model.addConstr(
                gp.quicksum(x[arc] for arc in outgoing_arcs) <= 1,
                name=f"sc_usage_{sc}"
            )

        # 3. Each WI can have at most one immediate successor
        for wi in WIs:
            outgoing_arcs = [arc for arc in A if arc[0] == wi]
            model.addConstr(
                gp.quicksum(x[arc] for arc in outgoing_arcs) <= 1,
                name=f"flow_{wi}"
            )

        # 4. Hard time window constraints
        for wi in WIs:
            # Must start within time window
            model.addConstr(t_s[wi] >= wi.earliest_move_start_time, name=f"ES_{wi}")
            model.addConstr(t_s[wi] <= wi.latest_move_start_time, name=f"LS_{wi}")

            # End time = start time + operation time
            model.addConstr(t_e[wi] == t_s[wi] + wi.operation_time, name=f"duration_{wi}")

            # Must end within time window
            model.addConstr(t_e[wi] >= wi.earliest_move_end_time, name=f"EE_{wi}")
            model.addConstr(t_e[wi] <= wi.latest_move_end_time, name=f"LE_{wi}")

        # 5. Time consistency for SC->WI arcs
        for arc in A:
            if isinstance(arc[0], type(SCs[0])):  # If first node is SC
                sc, wi = arc[0], arc[1]
                # WI can't start before SC is available + travel time
                model.addConstr(
                    t_s[wi] >= sc.start_time + d[arc] - M * (1 - x[arc]),
                    name=f"time_sc_{sc}_{wi}"
                )

        # 6. Time consistency for WI->WI arcs
        for arc in A:
            if isinstance(arc[0], type(WIs[0])) and isinstance(arc[1], type(WIs[0])):
                wi_i, wi_j = arc[0], arc[1]
                # Second WI can't start before first WI ends + travel time
                model.addConstr(
                    t_s[wi_j] >= t_e[wi_i] + d[arc] - M * (1 - x[arc]),
                    name=f"time_wi_{wi_i}_{wi_j}"
                )

        # Solve
        model.Params.TimeLimit = time_limit
        model.optimize()

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            # Extract solution
            solution_arcs = [(arc[0], arc[1]) for arc in A if x[arc].X > 0.5]

            # Build routes for each SC
            routes = {}
            for sc in SCs:
                route = []
                current = sc
                visited = set()

                while True:
                    next_node = None
                    for arc in solution_arcs:
                        if arc[0] == current and arc[1] not in visited:
                            next_node = arc[1]
                            break

                    if next_node is None:
                        break

                    route.append(next_node)
                    visited.add(next_node)
                    current = next_node

                routes[sc] = route

            # Calculate statistics
            total_travel = sum(d[arc] * x[arc].X for arc in A)
            total_loaded = sum(wi.operation_time for wi in WIs)

            # Update instance attributes
            self.objective_function = model.ObjVal
            self.best_found_objective_function = model.ObjVal

            # Update allocated work instructions as a list-of-lists, aligned with self.problem.scs
            self.allocated_work_instructions = [[] for _ in SCs]
            index_by_sc = {sc: i for i, sc in enumerate(SCs)}

            for sc, route in routes.items():
                i = index_by_sc[sc]
                # Save the route list for this SC
                self.allocated_work_instructions[i] = list(route)

                # Annotate WI objects
                for wi in route:
                    wi.sc = sc
                    wi.assigned = True
                    wi.planned_start_time = float(t_s[wi].X)
                    wi.planned_end_time = float(t_e[wi].X)

            return True

        else:
            print(f"No solution found. Status: {model.status}")
            return False



problem = Problem_setting.Problem(instance_name="Instances/finalV3_5SC_30WI_medium_20.csv")
solution = Solutions(problem)
# solution.solve_with_gurobi(time_limit=180)
# print(solution.objective_function)
# solution.print_allocations()
#

# value = solution.the_greedy()
# print(value)
# # # # print(solution.objective_function)
# # # # # # solution.print_allocations()
# solution.local_search()
# print(solution.objective_function)

solution.GRASP(time_limit=60)
print(solution.objective_function)
#
# solution.VNS(time_limit=60)
# print(solution.objective_function)

# assignment = """
# SC0 [ WI22 WI54 WI19 WI6 WI62 WI12 WI15 WI59 WI21 WI70 ]
# SC1 [ WI14 WI37 WI60 WI82 WI96 WI61 WI46 WI13 WI49 WI5 WI2 WI25 ]
# SC2 [ WI80 WI73 WI35 WI77 WI7 WI30 WI87 WI53 WI84 ]
# SC3 [ WI27 WI58 WI94 WI85 WI72 WI66 WI9 WI69 WI71 ]
# SC4 [ WI17 WI83 WI23 WI16 WI20 WI26 WI10 WI57 WI51 WI43 ]
# SC5 [ WI76 WI92 WI65 WI33 WI90 WI1 WI55 WI91 WI39 ]
# SC6 [ WI8 WI67 WI48 WI11 WI36 WI97 WI47 WI95 WI75 WI3 ]
# SC7 [ WI34 WI89 WI44 WI63 WI64 WI24 WI81 WI98 WI41 WI78 WI32 ]
# SC8 [ WI18 WI74 WI88 WI40 WI4 WI28 WI38 WI79 WI42 WI45 WI99 ]
# SC9 [ WI56 WI31 WI29 WI0 WI52 WI86 WI93 WI68 WI50 ]
# """
#
# is_feasible, objective = solution.set_manual_assignment(assignment)
# print(is_feasible, objective)
