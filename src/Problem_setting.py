import csv


class Location:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, location):
        return abs(self.x - location.x) + abs(self.y - location.y)

    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class SC:
    def __init__(self, name, index, start_position, start_time):
        self.name = name
        self.index = index
        self.start_position = start_position
        self.start_time = start_time
        self.current_position = start_position
        self.last_position = start_position
        self.last_WI_finish_time = None
        self.next_WI = None
        self.last_WI = None

    def __str__(self):
        return self.name


class WI:
    def __init__(self, name, index, origin_location, final_location, earliest_move_start_time,
                 latest_move_start_time, earliest_move_end_time, latest_move_end_time, operation_time):
        self.index = index
        self.name = name
        self.origin_location = origin_location
        self.final_location = final_location
        self.earliest_move_start_time = earliest_move_start_time
        self.latest_move_start_time = latest_move_start_time
        self.earliest_move_end_time = earliest_move_end_time
        self.latest_move_end_time = latest_move_end_time
        self.planned_start_time = None
        self.planned_end_time = None
        self.sc = None
        self.assigned = False
        self.operation_time = operation_time
        self.job_type = None

    def __str__(self):
        return self.name


class Problem:
    def __init__(self, instance_name):
        self.instance_name = instance_name
        self.wis = []
        self.scs = []
        self.wis_number = 0
        self.sc_number = 0
        self.horizon = 0
        self.durations = {}
        self.speed = 3

        self.read_instance(instance_name)

    def read_instance(self, instance_name):

        print(f"Reading instance from {instance_name}...")

        object_map = {}

        with open(instance_name, 'r') as f:
            reader = csv.reader(f)

            self.horizon = int(next(reader)[1])
            self.sc_number = int(next(reader)[1])
            self.wis_number = int(next(reader)[1])
            next(reader)  # Skip blank line

            header = next(reader)
            col_map = {name: i for i, name in enumerate(header)}

            for row in reader:
                if not row:
                    break

                obj_type = row[col_map["type"]]
                obj_id = row[col_map["id"]]

                if obj_type == "SC":
                    x = float(row[col_map["x"]])
                    y = float(row[col_map["y"]])
                    start_time = int(row[col_map["start_time"]])

                    start_pos = Location(x, y)
                    sc_obj = SC(
                        name=obj_id,
                        index=len(self.scs),
                        start_position=start_pos,
                        start_time=start_time
                    )

                    self.scs.append(sc_obj)
                    object_map[obj_id] = sc_obj

                elif obj_type == "WI":
                    o_x = float(row[col_map["o_x"]])
                    o_y = float(row[col_map["o_y"]])
                    d_x = float(row[col_map["d_x"]])
                    d_y = float(row[col_map["d_y"]])

                    origin_loc = Location(o_x, o_y)
                    dest_loc = Location(d_x, d_y)

                    wi_obj = WI(
                        name=obj_id,
                        index=len(self.wis),
                        origin_location=origin_loc,
                        final_location=dest_loc,
                        earliest_move_start_time=int(row[col_map["ES"]]),
                        latest_move_start_time=int(row[col_map["LS"]]),
                        earliest_move_end_time=int(row[col_map["EE"]]),
                        latest_move_end_time=int(row[col_map["LE"]]),
                        operation_time=int(row[col_map["op_time"]])
                    )

                    self.wis.append(wi_obj)
                    object_map[obj_id] = wi_obj

            next(reader)

            for row in reader:
                if not row:
                    continue

                source_id = row[0]
                dest_id = row[1]
                travel_time = float(row[2])

                source_obj = object_map[source_id]
                dest_obj = object_map[dest_id]

                if source_obj not in self.durations:
                    self.durations[source_obj] = {}

                # Populate the dictionary
                self.durations[source_obj][dest_obj] = travel_time / self.speed

        # print("Instance reading complete.")
        # print(f"  - Horizon: {self.horizon}")
        # print(f"  - Loaded {len(self.scs)} SCs and {len(self.wis)} WIs.")
        # print(f"  - Calculated {sum(len(v) for v in self.durations.values())} empty travel durations.")


# if __name__ == "__main__":
#     # Test with the generated instance
#     problem = Problem(instance_name="Instances/simple_5SC_60WI_extreme.csv")
#
#     # Show some sample data
#     print(f"\nFirst SC: {problem.scs[0].name} at {problem.scs[0].start_position}")
#     print(f"First WI: {problem.wis[0].name}")
#     print(f"  Origin: {problem.wis[0].origin_location}")
#     print(f"  Destination: {problem.wis[0].final_location}")
#     print(f"  Operation time: {problem.wis[0].operation_time}")
#     print(f"  Time window: [{problem.wis[0].earliest_move_start_time}, {problem.wis[0].latest_move_start_time}]")