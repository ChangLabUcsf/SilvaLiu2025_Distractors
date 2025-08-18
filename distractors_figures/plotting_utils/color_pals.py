# From : https://github.com/kevinsblake/NatParksPalettes/blob/main/R/NatParksPalettes.R

regions = ["#082544", "#1E547D", "#79668C", "#DE3C37", "#F2DC7E"][::-1]
distractors = ["#E07529", "#FAAE32", "#7F7991", "#A84A00", "#5D4F36", "#B39085"]
grey_scale = ["#000000", '#747474', '#CDCDCD']
# det_model_type = ["#293633", "#3D5051", "#6B7F7F", "#87A1C7", "#516B95", "#304F7D"]
det_model_type = ["#1D4A79", "#794C23", "#6B7444", "#6089B5", "#BF9785", "#275E4D", "#807B7F"]

region_dict = {'temporal': regions[0],
               'postcentral': regions[1],
               'frontal': regions[2],
               'precentral': regions[3],
               'all': regions[-1],
              }