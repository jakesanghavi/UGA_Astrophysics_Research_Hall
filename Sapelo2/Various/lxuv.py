from atm_mass_frac import evolve_atmosphere_test

xuvs = evolve_atmosphere_test(
                Mc_me=1,
                a_AU=1,
                t_disk_Myr=3.0,
                t_end_Gyr=5.0,
                init=0.15,
                dusty=True,
                eta=0.1,
                Lxuv0=2.05e22,
                t_sat_Myr=100,
                decay_index=1.1
            )

print(xuvs[-1])