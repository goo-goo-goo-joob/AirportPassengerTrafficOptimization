from main import main_spb, main_svo, main_spb_brute_force, main_svo_brute_force, main_synthetic, \
    main_synthetic_brute_force


def test_main_spb_SLSQP(benchmark):
    benchmark(main_spb, 'SLSQP')


def test_main_sbp_trust_constr(benchmark):
    benchmark(main_spb, 'trust-constr')


def test_main_spb_COBYLA(benchmark):
    benchmark(main_spb, 'COBYLA')


def test_main_svo_SLSQP(benchmark):
    benchmark(main_svo, 'SLSQP')


def test_main_svo_trust_constr(benchmark):
    benchmark(main_svo, 'trust-constr')


def test_main_svo_COBYLA(benchmark):
    benchmark(main_svo, 'COBYLA')


def test_main_spb_brute_force(benchmark):
    benchmark(main_spb_brute_force)


def test_main_svo_brute_force(benchmark):
    benchmark(main_svo_brute_force)


def test_main_synthetic_SLSQP_2(benchmark):
    benchmark(main_synthetic, 'SLSQP', 2)


def test_main_synthetic_trust_constr_2(benchmark):
    benchmark(main_synthetic, 'trust-constr', 2)


def test_main_synthetic_COBYLA_2(benchmark):
    benchmark(main_synthetic, 'COBYLA', 2)


def test_main_synthetic_SLSQP_3(benchmark):
    benchmark(main_synthetic, 'SLSQP', 3)


def test_main_synthetic_trust_constr_3(benchmark):
    benchmark(main_synthetic, 'trust-constr', 3)


def test_main_synthetic_COBYLA_3(benchmark):
    benchmark(main_synthetic, 'COBYLA', 3)


def test_main_synthetic_SLSQP_4(benchmark):
    benchmark(main_synthetic, 'SLSQP', 4)


def test_main_synthetic_trust_constr_4(benchmark):
    benchmark(main_synthetic, 'trust-constr', 4)


def test_main_synthetic_COBYLA_4(benchmark):
    benchmark(main_synthetic, 'COBYLA', 4)


def test_main_synthetic_SLSQP_5(benchmark):
    benchmark(main_synthetic, 'SLSQP', 5)


def test_main_synthetic_trust_constr_5(benchmark):
    benchmark(main_synthetic, 'trust-constr', 5)


def test_main_synthetic_COBYLA_5(benchmark):
    benchmark(main_synthetic, 'COBYLA', 5)


def test_main_synthetic_SLSQP_6(benchmark):
    benchmark(main_synthetic, 'SLSQP', 6)


def test_main_synthetic_trust_constr_6(benchmark):
    benchmark(main_synthetic, 'trust-constr', 6)


def test_main_synthetic_COBYLA_6(benchmark):
    benchmark(main_synthetic, 'COBYLA', 6)


def test_main_synthetic_SLSQP_7(benchmark):
    benchmark(main_synthetic, 'SLSQP', 7)


def test_main_synthetic_trust_constr_7(benchmark):
    benchmark(main_synthetic, 'trust-constr', 7)


def test_main_synthetic_COBYLA_7(benchmark):
    benchmark(main_synthetic, 'COBYLA', 7)


def test_main_synthetic_brute_force_2(benchmark):
    benchmark(main_synthetic_brute_force, 2)


def test_main_synthetic_brute_force_3(benchmark):
    benchmark(main_synthetic_brute_force, 3)


def test_main_synthetic_brute_force_4(benchmark):
    benchmark(main_synthetic_brute_force, 4)
