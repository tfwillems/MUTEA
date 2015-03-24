def read_stutter_info(input_file, min_pgeom=0.0):
    stutter_info = {}
    data = open(input_file, "r")
    for line in data:
        tokens = line.strip().split()
        if tokens[7] != "CONVERGED" and tokens[7] != "ESTIMATED_FROM_SEQUENCE":
            continue
        if float(tokens[4]) < min_pgeom:
            continue
        stutter_info[tokens[0]+":"+tokens[1]+"-"+tokens[2]] = {"P_GEOM":float(tokens[4]), "DOWN":float(tokens[5]), "UP":float(tokens[6])}
    data.close()
    return stutter_info
