start_string_mos_file = """
#1
double tab1(8760,30)
#LOCATION,Potsdam,State,Country,TMY3,Something,52.23611890272718,13.039320273252379,1.0,Something
#DESIGN CONDITIONS,BlaBla
#TYPICAL/EXTREME PERIODS,BlabBla
#GROUND TEMPERATURES, BlaBla
#HOLIDAYS/DAYLIGHT SAVINGS,BlaBla
#COMMENTS BlaBla
#DATA PERIODS,BlaBla
#C1 Time in seconds. Beginning of a year is 0s.
#C2 Dry bulb temperature in Celsius at indicated time
#C3 Dew point temperature in Celsius at indicated time
#C4 Relative humidity in percent at indicated time
#C5 Atmospheric station pressure in Pa at indicated time
#C6 Extraterrestrial horizontal radiation in Wh/m2
#C7 Extraterrestrial direct normal radiation in Wh/m2
#C8 Horizontal infrared radiation intensity in Wh/m2
#C9 Global horizontal radiation in Wh/m2
#C10 Direct normal radiation in Wh/m2
#C11 Diffuse horizontal radiation in Wh/m2
#C12 Averaged global horizontal illuminance in lux during minutes preceding the indicated time
#C13 Direct normal illuminance in lux during minutes preceding the indicated time
#C14 Diffuse horizontal illuminance in lux  during minutes preceding the indicated time
#C15 Zenith luminance in Cd/m2 during minutes preceding the indicated time
#C16 Wind direction at indicated time. N=0, E=90, S=180, W=270
#C17 Wind speed in m/s at indicated time
#C18 Total sky cover at indicated time
#C19 Opaque sky cover at indicated time
#C20 Visibility in km at indicated time
#C21 Ceiling height in m
#C22 Present weather observation
#C23 Present weather codes
#C24 Precipitable water in mm
#C25 Aerosol optical depth
#C26 Snow depth in cm
#C27 Days since last snowfall
#C28 Albedo
#C29 Liquid precipitation depth in mm at indicated time
#C30 Liquid precipitation quantity
"""
import pandas as pd
import pathlib
from ebcpy.preprocessing import convert_datetime_index_to_float_index


def get_vals(idx, df):
    map_columns = {
        1: "time",   #C1 Time in seconds. Beginning of a year is 0s.
        2: "Glasgow Temperature [2 m elevation corrected]",    #C2 Dry bulb temperature in Celsius at indicated time
        3: "Glasgow Temperature [2 m elevation corrected]",    #C3 Dew point temperature in Celsius at indicated time
        4: "Glasgow Relative Humidity [2 m]",    #C4 Relative humidity in percent at indicated time
        5: 101325.0,    #C5 Atmospheric station pressure in Pa at indicated time
        8: 0,    #C8 Horizontal infrared radiation intensity in Wh/m2
        9: "Glasgow Shortwave Radiation",    #C9 Global horizontal radiation in Wh/m2
        10: "Glasgow Direct Shortwave Radiation",   #C10 Direct normal radiation in Wh/m2
        11: "Glasgow Diffuse Shortwave Radiation",   #C11 Diffuse horizontal radiation in Wh/m2
        16: "Glasgow Wind Direction [10 m]",   #C16 Wind direction at indicated time. N=0, E=90, S=180, W=270
        17: "Glasgow Wind Speed [10 m]",   #C17 Wind speed in m/s at indicated time
        18: 0    #C18 Total sky cover at indicated time
    }
    default_value = 99999.9  # Value for mos file if no value is matching
    val = map_columns.get(idx, default_value)
    if isinstance(val, str):
        val = df[val]
    return str(val)


def convert_xsl_to_mos(path_xsl, sheet_name_data="data_only", sep="\t"):

    df = pd.read_excel(path_xsl, sheet_name=sheet_name_data)
    df = df.set_index("timestamp")
    df = convert_datetime_index_to_float_index(df)
    df["time"] = df.index
    output = start_string_mos_file
    for time_idx in range(len(df["time"])):
        output += sep.join([get_vals(col_idx, df.iloc[time_idx]) for col_idx in range(30)]) + "\n"

    with open(pathlib.Path(path_xsl).parent.joinpath("Glasgow_data.mos"), "w+") as file:
        file.write(output)


if __name__ == "__main__":
    convert_xsl_to_mos(path_xsl=r"D:\00_temp\extreme climate_glasgow.xlsx")