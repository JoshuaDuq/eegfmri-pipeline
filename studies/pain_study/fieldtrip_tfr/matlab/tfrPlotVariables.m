function variables = tfrPlotVariables()
% tfrPlotVariables Return every saved TFR plot and its scale type.
    names = [
        "pow_temp_44p3"
        "pow_temp_44p3_db"
        "pow_temp_45p3"
        "pow_temp_45p3_db"
        "pow_temp_46p3"
        "pow_temp_46p3_db"
        "pow_temp_47p3"
        "pow_temp_47p3_db"
        "pow_temp_48p3"
        "pow_temp_48p3_db"
        "pow_temp_49p3"
        "pow_temp_49p3_db"
        "pow_painful"
        "pow_painful_db"
        "pow_nonpainful"
        "pow_nonpainful_db"
        "pow_painful_v_nonpainful"
        "pow_low_temperature"
        "pow_high_temperature"
        "pow_high_v_low"
        "pow_temperature_slope"
        "pow_avg"
    ];
    signed = [
        false
        true
        false
        true
        false
        true
        false
        true
        false
        true
        false
        true
        false
        true
        false
        true
        true
        false
        false
        true
        true
        false
    ];
    variables = table(names, signed);
end
