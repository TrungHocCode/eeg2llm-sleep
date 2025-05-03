# # Giá trị nhãn

# W = 0
# N1 = 1
# N2 = 2
# N3 = 3
# REM = 4
# MOVE = 5
# UNK = 6
# EPILEPSY = 7
# SLEEP_DISORDER = 8
# STROKE = 9

# stage_dictionary = {
#     "W" : W,
#     "N1" : N1,
#     "N2" : N2,
#     "N3" : N3,
#     "REM" : REM,
#     "MOVE" : MOVE,
#     "UNK" : UNK,
# }

# class_dictionary = {
#     W: "W",
#     N1: "N1",
#     N2: "N2",
#     N3: "N3",
#     REM: "REM",
#     MOVE: "MOVE",
#     UNK: "UNK",
#     EPILEPSY: "Epilepsy",
#     SLEEP_DISORDER: "Sleep Disorder",
#     STROKE: "Stroke",
# }

stage_dictionary = {
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "MOVE": 5,
}

class_dictionary = {
    0: "Sleep stage W",
    1: "Sleep stage N1",
    2: "Sleep stage N2",
    3: "Sleep stage N3",
    4: "Sleep stage R",
    5: "Obstructive Apnea",
    6: "Obstructive Hypopnea",
    7: "Mixed Apnea",
    8: "Central Apnea",
    9: "Oxygen Desaturation",
    10: "EEG arousal",
    11: "Hypopnea"
}
