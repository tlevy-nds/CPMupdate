library(mice)
library(dplyr)
library(datetime)
library(tidyr)
library(data.table)

# I originally wanted to keep labs that were integers as integers but couldn't get it to work so I made them all doubles
is.wholenumber <- function(x, tol = .Machine$double.eps^0.5)  abs(x - round(x)) < tol

substrLeft <- function(x, n){substr(x, 1, nchar(x)-n)}

# load the data
filename = r"(FULLPATH\COVID_reduced_Spring_2022.csv)"
data <- read.csv(filename, header=TRUE, sep=",")  #, colClasses=colClasses)

# convert fields that end with Dtm to datetime
data <- data %>% as_tibble() %>% mutate(across(ends_with("Dtm"), as.datetime))

# order O2_Delivery
o2deliveryLevels <- c("room air", "nasal cannula", "mask", "nonrebreather", "nc + nrb",
                      "high flow nasal cannula", "bipap/cpap", "nc + bipap", "ventilator", "ecmo",
                      "ecmo + ventilator")
data$O2_Delivery <- factor(data$O2_Delivery, levels=o2deliveryLevels)

# categorical variables need to be factors
catVars = c("FinalHospital", "Gender", "Race", 
            "Ethnicity", "Language", 
            "ABO_Interpretation", "ABO_Rh_Confirmation", "ABO_RH_Interpretation")
data <- mutate(data, across(any_of(catVars), ~factor(.x, exclude=c("NA"))))


vitals = c("SBP", "DBP", "HR", "RR", "Temp", "SPO2")
labs = c("Alanine_Aminotransferase_.ALT.SGPT.", "Albumin_Serum", "Alkaline_Phosphatase_Serum", 
         "Aspartate_Aminotransferase_.AST.SGOT.", "Auto_Eosinophil_.", "Auto_Eosinophil_..1", 
         "Auto_Lymphocyte_.", "Auto_Lymphocyte_..1", "Auto_Monocyte_.", "Auto_Monocyte_..1", 
         "Auto_Neutrophil_.", "Auto_Neutrophil_..1", "Bilirubin_Total_Serum", "Blood_Urea_Nitrogen_Serum", 
         "Carbon_Dioxide_Serum", "Chloride_Serum", "Creatinine_Serum", "DDimer_Assay_Quantitative", 
         "Ferritin_Serum", "Glucose_Serum", "Hemoglobin", "Lactate", "Lactate_Dehydrogenase_Serum", 
         "pCO2_Arterial", "pH_Arterial", "Platelet_Count__Automated", "pO2_Arterial", 
         "Potassium_Serum", "Procalcitonin_Serum", "Prothrombin_Time_Plasma", "Red_Cell_Distrib_Width", 
         "Sodium_Serum", "Troponin_I_Serum", "Troponin_T_High_Sensitivity", "Troponin_T_High_Sensitivity_Result", 
         "Troponin_T_Serum", "WBC_Count", "eGFR", "CRP")

data <- mutate(.data=data, 
               los = (data$DischargeDtm - data$AdmitDtm) / 3600,
               .keep="unused")  # .keep is experimental and isn't working for me

# some of the lab values are text so convert all labs to double
data <- mutate(data, across(any_of(labs), as.double))  # if_else(all(is.wholenumber, na.rm=TRUE), as.integer, as.double)

# create log labs
data <- mutate(data, across(any_of(labs), ~log(replace(.x, .x == 0, min(.x[.x > 0], na.rm=TRUE))), .names="{col}_log"))

# list of columns not used in the imputation models
excludeList <- c("Index", "AdmitDtm","DischargeDtm", "DeceasedDtm",
                 "IsTransfer", "OutsideTransfer", 
                 "VentDtm", 
                 "DNRDtm")
ini <- mice(data, maxit=0, print=F)
meth <- ini$method
pred <- ini$predictorMatrix
# pred <- quickpred(data, minpuc=0.5, exclude=excludeList)

# passive imputation for BMI (sum(is.na(data$BMI)) < sum(!is.na(data$BMI)))
data$Height[which(data$Height == 0)] <- NA
data$Height[which(data$Weight == 0)] <- NA
meth["BMI"] <- "~ I(Weight / (Height / 100)^2)"
pred[c("Height", "Weight"), "BMI"] <- 0

# passive imputation for _log
df = data.frame(meth)
setDT(df, keep.rownames = TRUE)[]
for(name in df$rn) {
  if(endsWith(name, "_log")) {
    ind <- which(df$rn == name)
    origName = substr(name, 1, nchar(name) - 4)
    # ind2 <- which(df$rn == origName)
    df$meth[ind] <- paste("~ I(log(", origName, "))", sep="")
    pred[origName, name] <- 0
  }
}

pred[excludeList, ] <- 0
pred[ ,excludeList] <- 0

inds <- -match(excludeList, names(data))
data2 <- data[ , inds]
pred2 <- pred[inds, inds]
meth2 <- meth[inds]
devCutoff = as.datetime("2020-04-23T00:00:00")
ignore <- data$AdmitDtm >= devCutoff

m <- 5
P <- ncol(data2)

# A condition number becomes small before 5 iterations, so keep trying until the imputation runs successfully.
ct <- 0
while(ct < 5)
{
  out <- tryCatch(
  {
    ct <- 0
    imp <- mice(data2, m=m, maxit=1, pred=pred2, meth=meth2, nnet.MaxNWts=2000, ignore=ignore, print=T)
    ct <- 1
    for(it2 in 2:5) {
      imp <- mice.mids(imp, newdata=NULL, maxit=1, printFlag=TRUE)  # , m=m, pred=pred2, meth=meth2, nnet.MaxNWts=2000, ignore=ignore)  # , seed=123
      ct <- ct + 1
    }
  },
  error=function(cond) {
    ct <- 0
  },
  finally={
    # pass
  })
}


for(im in 1:m) {
  write.table(complete(imp, im), file=paste(r"(FULLPATH\COVID_reduced_Spring_2022_imp_new)", im, ".csv", sep=""), sep=",")
}