def standardize_asllex_vocab(gloss: str) -> str:
    pass
    # Weird things in the ASL Knowledge Graph
    # Non-alphanumeric characters for asllex:#9_oclock: {'#'}
    # Non-alphanumeric characters for asllex:walk-tightrope-cl: {'-'}
    # Non-alphanumeric characters for asllex:#5_dollars: {'#'}
    # Non-alphanumeric characters for asllex:#1_dollar: {'#'}
    # Non-alphanumeric characters for asllex:think_chin.m4v: {'.'}
    # Non-alphanumeric characters for asllex:splash-ca: {'-'}
    # Non-alphanumeric characters for asllex:ski-ca: {'-'}
    # Non-alphanumeric characters for asllex:W.H.A.T: {'.'}
    # Non-alphanumeric characters for asllex:w.h.a.t: {'.'}
    # Non-alphanumeric characters for asllex:toothbrush-ca: {'-'}
    # Non-alphanumeric characters for asllex:this/it: {'/'}
    # Non-alphanumeric characters for asllex:release,_rescue: {','}
    # Non-alphanumeric characters for asllex:stretch-ca: {'-'}
    # Non-alphanumeric characters for asllex:#8_hour: {'#'}
    # Non-alphanumeric characters for asllex:a-line_bob: {'-'}
    # Non-alphanumeric characters for asllex:release, rescue: {' ', ','}
    # Non-alphanumeric characters for don't_feel_like: {"'"}
    # Non-alphanumeric characters for work_out_(exercise): {')', '('}
    # Non-alphanumeric characters for rotisserie_(to_roast_over_a_fire): {')', '('}
    # Complete set: {'/', '-', '#', ',', '.'}

    # ASL Citizen Test Set has 2731 vocab
    # Non-alphanumeric characters for THIS/IT: {'/'}
    # Non-alphanumeric characters for SKI-CA: {'-'}
    # Non-alphanumeric characters for WALK-TIGHTROPE-CL: {'-'}
    # Non-alphanumeric characters for HURDLE/TRIP1: {'/'}
    # Non-alphanumeric characters for W.H.A.T: {'.'}
    # Non-alphanumeric characters for STRETCH-CA: {'-'}
    # Non-alphanumeric characters for A-LINEBOB: {'-'}
    # Non-alphanumeric characters for TOOTHBRUSH-CA: {'-'}
    # Non-alphanumeric characters for SPLASH-CA: {'-'}
    # Non-alphanumeric characters for STAND-UP: {'-'}
    # Non-alphanumeric characters for HURDLE/TRIP2: {'/'}
    # Complete set: {'/', '-', '.'}

    # Sem-Lex
    # Quite a few, including:
    # Non-alphanumeric characters for lantes?: {'?'}
    # Non-alphanumeric characters for 5_o'clock: {"'"}
    # Non-alphanumeric characters for cant-see: {'-'}
    # Non-alphanumeric characters for do?: {'?'}
    # Non-alphanumeric characters for rudolph_the reindeer: {' '}
    # Non-alphanumeric characters for don't_understand: {"'"}
    # Non-alphanumeric characters for f√©minine: {'√', '©'}
    # Non-alphanumeric characters for monday tuesday wednesday thursday: {' '}
    # Non-alphanumeric characters for cl:3: {':'}
    # Non-alphanumeric characters for 75%_more: {'%'}
    # Non-alphanumeric characters for 11-jul: {'-'}
    # Non-alphanumeric characters for  tsk: {' '}
    # Non-alphanumeric characters for ?: {'?'}
    # Non-alphanumeric characters for @: {'@'}
    # Non-alphanumeric characters for  cook: {' '}
    # Complete set: {'√', '©', '!', '(', '?', ' ', '-', '@', '.', ')', '/', '%', "'", ':', ','}

    # ASL Lex 2.0 has 2719 vocab
    # Non-alphanumeric characters for W.H.A.T: {'.'}
    # Non-alphanumeric characters for release, rescue: {',', ' '}
    # Non-alphanumeric characters for murder : {' '}
    # Non-alphanumeric characters for ski-ca: {'-'}
    # Non-alphanumeric characters for this/it: {'/'}
    # Non-alphanumeric characters for santa  : {' '}
    # Non-alphanumeric characters for toothbrush-ca: {'-'}
    # Non-alphanumeric characters for splash-ca: {'-'}
    # Non-alphanumeric characters for walk-tightrope-cl: {'-'}
    # Non-alphanumeric characters for a-line_bob: {'-'}
    # Non-alphanumeric characters for fry : {' '}
    # Non-alphanumeric characters for stretch-ca: {'-'}
    # Non-alphanumeric characters for close : {' '}
    # Complete set: {',', '/', '-', '.', ' '}


def test_gloss_matching():

    # ASL Citizen Test Set has 2731 vocab
    # Non-alphanumeric characters for STAND-UP: {'-'}
    # Non-alphanumeric characters for SKI-CA: {'-'}
    # Non-alphanumeric characters for SPLASH-CA: {'-'}
    # Non-alphanumeric characters for A-LINEBOB: {'-'}
    # Non-alphanumeric characters for TOOTHBRUSH-CA: {'-'}
    # Non-alphanumeric characters for STRETCH-CA: {'-'}
    # Non-alphanumeric characters for THIS/IT: {'/'}
    # Non-alphanumeric characters for W.H.A.T: {'.'}
    # Non-alphanumeric characters for HURDLE/TRIP2: {'/'}
    # Non-alphanumeric characters for HURDLE/TRIP1: {'/'}
    # Non-alphanumeric characters for WALK-TIGHTROPE-CL: {'-'}

    test_tuples = [
        (
            "asllex:#9_oclock",  # ASL Knowledge Graph
            "9_oclock",  # ASL Lex 2.0
            "9OCLOCK",  # ASL Citizen test set
        ),
        (
            "OH I SEE",
            "asllex:oh_I_see",  # ASL Knowledge Graph
            "oh_i_see",  # SemLex
            "oh_I_see",  # yes, with the capital I, from ASL Lex 2.0
            "OHISEE",  # ASL Citizen
        ),
        ("think_chin", "think chin", "think_chin.m4v", "THINKCHIN"),
        ("release, rescue", "release,_rescue", "release,rescue", "RELEASERESCUE"),
        ("W.H.A.T", "w.h.a.t"),  # https://asl-lex.org/visualization/viewdata.html G_02_089
        ("ALL GONE", "ALLGONE"),
        # "don't_have_any", #ASLKG
        # don't_understand # ASLKG
        (
            "DONTFEELLIKE",  # ASL Citizen, also J_01_005
            "dont_feel_like",  # ASL LEX 2.0 entry for J_01_005
            "don't_feel_like",  # sem-lex dont_feel_like
            "asllex:dont_feel_like",  # ASLKG
        ),
        (
            "STAND-UP",  # ASL Citizen lists J_03_070
            "stand_up"  # ASL Lex 2.0
            "stand_up",  # Sem-Lex
            "asllex:stand_up",  # ASLKG "response" relation
        ),
        (
            "a-line_bob",  # ASL Lex 2.0
            "asllex:a-line_bob",  # ASLKG
            "A-LINEBOB",  # ASL Citizen
        ),
        (
            "HURDLE/TRIP1",  # ASL Citizen Test Set, which lists G_01_036
            "hurdle_2,hurdle/trip",  # Note the 2! hurdle_2,hurdle/trip,G_01_036 ASL Lex 2.0
            "asllex:hurdle_2",
        ),
        (
            "HURDLE/TRIP2",  # ASL Citizen Test Set: HURDLE/TRIP2,F_02_012
            "hurdle_1",  # ASL Lex 2.0 hurdle_1,hurdle/trip,F_02_012
            "asllex:hurdle_1",  # ASLKG
        ),
        (
            "I_love_you", # ASL-LEX 2 EntryID (F_01_102)
            "i_love_you", # ASL-LEX 2 LemmaID, Sem-Lex 'label' (F_01_102)
            "ILOVEYOU", #ASL Citizen test set (F_01_102)
            "asllex:i_love_you", # ASLKG
            
        )
    ]
