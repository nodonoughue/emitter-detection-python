import make_figures

close_figs = True
force_recalc = True

do_book_1 = True
do_book_2 = True

if do_book_1:
    print('************************************************************************************************')
    print('Generating all figures from ''Emitter Detection and Geolocation for Electronic Warfare'', 2019.')
    print('************************************************************************************************')
    print('close_figs = {}'.format(close_figs))
    print('force_recalc = {}'.format(force_recalc))

    print('***Chapter 1 ***')
    make_figures.chapter1.make_all_figures(close_figs=close_figs)
    print('***Chapter 2 ***')
    make_figures.chapter2.make_all_figures(close_figs=close_figs)
    print('***Chapter 3 ***')
    make_figures.chapter3.make_all_figures(close_figs=close_figs)
    print('***Chapter 4 ***')
    make_figures.chapter4.make_all_figures(close_figs=close_figs)
    print('***Chapter 5 ***')
    make_figures.chapter5.make_all_figures(close_figs=close_figs)
    print('***Chapter 6 ***')
    make_figures.chapter6.make_all_figures(close_figs=close_figs)
    print('***Chapter 7 ***')
    make_figures.chapter7.make_all_figures(close_figs=close_figs, force_recalc=force_recalc)
    print('***Chapter 8 ***')
    make_figures.chapter8.make_all_figures(close_figs=close_figs, force_recalc=force_recalc)
    print('***Chapter 9 ***')
    make_figures.chapter9.make_all_figures(close_figs=close_figs)
    print('***Chapter 10 ***')
    make_figures.chapter10.make_all_figures(close_figs=close_figs, force_recalc=force_recalc)
    print('***Chapter 11 ***')
    make_figures.chapter11.make_all_figures(close_figs=close_figs, force_recalc=force_recalc)
    print('***Chapter 12 ***')
    make_figures.chapter12.make_all_figures(close_figs=close_figs, force_recalc=force_recalc)
    print('***Chapter 13 ***')
    make_figures.chapter13.make_all_figures(close_figs=close_figs, force_recalc=force_recalc)
    print('***Appendix B ***')
    make_figures.appendixB.make_all_figures(close_figs=close_figs)
    print('***Appendix C ***')
    make_figures.appendixC.make_all_figures(close_figs=close_figs)
    print('***Appendix D ***')
    make_figures.appendixD.make_all_figures(close_figs=close_figs)
    print('Figure generation complete.')
else:
    print('************************************************************************************************')
    print('Skipping ''Emitter Detection and Geolocation for Electronic Warfare'', 2019.')
    print('Re-run with ''do_book_1 = True'' to generate all figures.')
    print('************************************************************************************************')

if do_book_2:
    print('************************************************************************************************')
    print('Generating all figures from ''Practical Geolocation for Electronic Warfare using MATLAB'', 2022.')
    print('************************************************************************************************')
    print('close_figs = {}'.format(close_figs))
    print('force_recalc = {}'.format(force_recalc))

    print('*** Chapter 1 ***')
    make_figures.practical_geo.chapter1.make_all_figures(close_figs=close_figs)
    print('*** Chapter 2 ***')
    make_figures.practical_geo.chapter2.make_all_figures(close_figs=close_figs, force_recalc=force_recalc)
    print('*** Chapter 3 ***')
    make_figures.practical_geo.chapter3.make_all_figures(close_figs=close_figs, force_recalc=force_recalc)
    print('*** Chapter 4 ***')
    make_figures.practical_geo.chapter4.make_all_figures(close_figs=close_figs, force_recalc=force_recalc)
    print('Figure generation complete.')
else:
    print('************************************************************************************************')
    print('Skipping ''Practical Geolocation for Electronic Warfare using MATLAB'', 2022.')
    print('************************************************************************************************')
    print('Re-run with ''do_book_1 = True'' to generate all figures.')