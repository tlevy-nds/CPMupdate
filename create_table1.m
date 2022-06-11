function [DataCell] = create_table1(r1, interval, outcomes, horizonDays, admitBefore, varargin)
p = inputParser();
p.addRequired('r1', @(x) isa(x, 'Reduced'));
p.addRequired('interval', @(x) length(x) == 2 && isdatetime(x));
p.addRequired('outcomes', @(x) iscellstr(x) && all(ismember(x, {'Expired_Outcome', 'Vented'})));
p.addRequired('horizonDays', @(x) isvector(x) && isnumeric(x));
p.addOptional('admitBefore', true, @(x) islogical(x) && isscalar(x));
p.addOptional('WordFileName', '', @(x) ischar(x));
p.parse(r1, interval, outcomes, horizonDays, admitBefore, varargin{:});

[r1, interval, outcomes, horizonDays, admitBefore, WordFileName] = deal(p.Results.r1, ...
     p.Results.interval, p.Results.outcomes, p.Results.horizonDays, p.Results.admitBefore, p.Results.WordFileName);

includeNotYetDischarged = true;
dateshift = days(5);

%% Header
myInds = [];
header = {'' 'All Included Patients'};
for ioutcome = 1:length(outcomes)
    outcome = outcomes{ioutcome};
    
    for ihorizon = 1:length(horizonDays)
        horizon = horizonDays(ihorizon);
        
        [~, ~, ~, ~, inds1, inds2] = r1.training_and_validation_set( ...
            includeNotYetDischarged, horizon * 24, {''}, interval(1), interval(2), Inf, [], admitBefore, outcome);

        % Manual correction
        % inds1(30008) = false;
        % inds2(30008) = false;

        if ioutcome == 1 && ihorizon == 1
            myInds = {inds1 | inds2};
        end
        myInds = [myInds {inds1 inds2}];
        
        switch outcome
            case 'Expired_Outcome'
                if isfinite(horizon)
                    header = [header {sprintf('Alive %d Days', horizon) sprintf('Died %d Days', horizon)}];
                else
                    header = [header {'Alive' 'Died'}];
                end
            case 'Vented'
                if isfinite(horizon)
                    header = [header {sprintf('Not Vented %d Days', horizon) sprintf('Vented %d Days', horizon)}];
                else
                    header = [header {'Not Vented' 'Vented'}];
                end
            otherwise
                error('unrecognized outcome');
        end
    end
end
header = [header {'Missing No. (%)'}];

DataCell = header;

%% Table entries
DataCell = [DataCell; {'n'} cellfun(@(x) num2str(nnz(x)), myInds, 'UniformOutput', false) {''}];

% gender
genderCats = categories(r1.ds.Gender);
for igender = 1:length(genderCats)
    genderCat = genderCats{igender};
    DataCell = [DataCell; row_n_pct(r1.ds.Gender, myInds, genderCat)];
end

% age
DataCell = [DataCell; {'Age, y (%)'} repmat({''}, [1, length(myInds) + 1])];
ages = categorical(1 * (r1.ds.Age >= 18 & r1.ds.Age <= 40) + ...
    2 * (r1.ds.Age >= 41 & r1.ds.Age <= 60) + ...
    3 * (r1.ds.Age >= 61 & r1.ds.Age <= 80) + ...
    4 * (r1.ds.Age >= 81 & r1.ds.Age <= 106), 1:4, {'18-40', '41-60', '61-80', '81-106'});
ageCats = categories(ages);
for iage = 1:length(ageCats)
    ageCat = ageCats{iage};
    DataCell = [DataCell; row_n_pct(ages, myInds, ageCat)];
end

% race
DataCell = [DataCell; {'Race (%)'} repmat({''}, [1, length(myInds) + 1])];
raceCats = categories(r1.ds.Race);
for irace = 1:length(raceCats)
    raceCat = raceCats{irace};
    DataCell = [DataCell; row_n_pct(r1.ds.Race, myInds, raceCat)];
end

% ethnicity
DataCell = [DataCell; {'Ethnicity (%)'} repmat({''}, [1, length(myInds) + 1])];
ethnicityCats = categories(r1.ds.Ethnicity);
for iethnicity = 1:length(ethnicityCats)
    ethnicityCat = ethnicityCats{iethnicity};
    DataCell = [DataCell; row_n_pct(r1.ds.Ethnicity, myInds, ethnicityCat)];
end

% English as a primary language
primaryLanguage = categorical(r1.ds.Language == 'ENGLISH', 0:1, {'Not English', 'English'});
DataCell = [DataCell; row_n_pct(primaryLanguage, myInds, 'English')];

% length of stay
DataCell = [DataCell; row_med_iqr(r1.los / 24, myInds, 'Length of stay, days (median [IQR])')];

% requires mechanic ventilation
vented = categorical(r1.ds.Vented, 0:1, {'Not Vented', 'Vented'});
DataCell = [DataCell; row_n_pct(vented, myInds, 'Vented')];

% Vitals
DataCell = [DataCell; {'Last emergency department vital sign measurement (median [IQR])'} repmat({''}, [1, length(myInds) + 1])];
myVars = {'SBP' 'DBP' 'HR' 'RR' 'Temp' 'SPO2', 'BMI' 'Height', 'Weight'};
myLabels = {'Systolic blood pressure, mmHg', 'Diastolic blood pressure, mmHg', 'Heart rate, beats per minute', ...
    'Respiratory rate, breaths per minute', 'Temperature, Celsius', 'Oxygen saturation, %', 'Body mass index, kg/m2', ...
    'Height, cm', 'Weight, kg'};
assert(length(myVars) == length(myLabels));
for ivital = 1:length(myVars)
    DataCell = [DataCell; row_med_iqr(r1.ds.(myVars{ivital}), myInds, myLabels{ivital})];
end

% Comorbidities
DataCell = [DataCell; {'Comorbidities, %'} repmat({''}, [1, length(myInds) + 1])];
myVars = {'CAD' 'DM' 'HTN' 'HF' 'LungDisease' 'KidneyDisease'};
myLabels = {'Coronary artery disease' 'Diabetes' 'Hypertension' 'Heart failure' 'Lung disease' 'Kidney disease'};
assert(length(myVars) == length(myLabels));
for icomorbidity = 1:length(myVars)
    myVar = myVars{icomorbidity};
    data = r1.ds.(myVar);
    co = categorical(data, 0:1, {'Not', myLabels{icomorbidity}});
    DataCell = [DataCell; row_n_pct(co, myInds, myLabels{icomorbidity})];
end

% Labs
DataCell = [DataCell; {'Last emergency department laboratory result (median [IQR])'} repmat({''}, [1, length(myInds) + 1])];
myVars = {'WBC_Count', 'Auto_Neutrophil_', 'Auto_Neutrophil_1', 'Auto_Lymphocyte_', 'Auto_Lymphocyte_1', ...
    'Auto_Eosinophil_', 'Auto_Eosinophil_1', 'Auto_Monocyte_', 'Auto_Monocyte_1', 'Hemoglobin', ...
    'Red_Cell_Distrib_Width', 'Platelet_Count__Automated', 'Sodium_Serum', 'Potassium_Serum', 'Chloride_Serum', ...
    'Carbon_Dioxide_Serum', 'Blood_Urea_Nitrogen_Serum', 'Creatinine_Serum', 'eGFR', 'Glucose_Serum', ...
    'Albumin_Serum', 'Bilirubin_Total_Serum', 'Alkaline_Phosphatase_Serum', 'Alanine_Aminotransferase_ALTSGPT', ...
    'Aspartate_Aminotransferase_ASTSGOT', 'CRP'};
myLabels = {'White blood cell count, K/µL', 'Absolute neutrophil, No., K/µL', 'Automated neutrophil, %', 'Automated lymphocyte, No., K/µL', ...
    'Automated lymphocyte, %', 'Automated eosinophil, No., K/µL', 'Automated eosinophil, %', 'Automated monocyte, No., K/µL', ...
    'Automated monocyte, %', 'Hemoglobin, g/dL', 'Red cell distribution width, %', 'Automated platelet count, K/µL', 'Serum sodium, mmol/L', ...
    'Serum potassium, mmol/L', 'Serum chloride, mmol/L', 'Serum carbon dioxide, mmol/L', 'Serum blood urea nitrogen, mg/dL', ...
    'Serum creatinine, mg/dL', 'eGFR mL/min/1.73m2', ...
    'Serum glucose, mg/dL', 'Serum albumin, g/dL', 'Total serum bilirubin, mg/dL', 'Serum alkaline phosphatase,U/L', ...
    'Alanine aminotransferase (ALT/SGPT), U/L', 'Aspartate aminotransferase (AST/SGOT), U/L', 'Serum C-Reactive Protein, mg/L'};
assert(length(myVars) == length(myLabels));
for ilab = 1:length(myVars)
    myVar = myVars{ilab};
    data = r1.ds.(myVar);    
    DataCell = [DataCell; row_med_iqr(data, myInds, myLabels{ilab})];
end

%% Write the table to a word document
if exist('WordFileName', 'var') && ~isempty(WordFileName)    
    write_table1(DataCell, WordFileName)
end


function myRow = row_n_pct(data, myInds, myCat)
missingCt = nnz(isundefined(data(myInds{1})));
N = numel(isundefined(data(myInds{1})));
myRow = [{sprintf('%s (%%)', myCat)} cellfun(@(x) sprintf('%d (%2.1f)', nnz(data(x) == myCat), ...
        100 * nnz(data(x) == myCat) / nnz(x)), myInds, 'UniformOutput', false) {sprintf('%d (%2.1f)', missingCt, 100 * missingCt / N)}];


function myRow = row_med_iqr(data, myInds, myRowName)
missingCt = nnz(isnan(data(myInds{1})));
N = numel(isnan(data(myInds{1})));
myRow = [{sprintf('%s', myRowName)} cellfun(@(x) sprintf('%2.2f [%3.2f, %3.2f]', ...
    quantile(data(x), [0.5 0.25 0.75])), myInds, 'UniformOutput', false) {sprintf('%d (%2.1f)', missingCt, 100 * missingCt / N)}];
