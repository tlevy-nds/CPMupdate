function create_mat_file(ds, impFiles, horizons)
% Usage: create_mat_file(ds, arrayfun(@(x) sprintf('COVID_reduced_May_2022_imp%d.csv', x), 1:5, 'UniformOutput', false), [28 7]*24);
% ds is the original data as a table
% impFiles is a cell array of filename strings for the imputed datasets
% horizon is in hours (e.g. 4*7*24 is 4 weeks)

% These variables will be excluded as predictor variables
excludeVars = {'ClientVisitGUID', 'EMPI', 'AdmitDtm', 'DischargeDtm', 'DeceasedDtm', 'Expired_Outcome', 'VisitStatus', 'Dispo', ...
                    'IsTransfer', 'OutsideTransfer', 'TransferDtm', 'FacilityCode', 'FinalHospital', ...
                    'CareLevelCode', 'Service', 'ServiceGroup', 'LACE_Sum', 'CCI', 'CvdResultDtm', 'HasIslStatus', 'Vented', 'VentDtm', ...
                    'AdmitToVent_hrs', 'TriageAcuity', 'ESIRecordedDtm', 'AdmitOrderDtm', 'ED_ArrivalDtm', 'ComfortOnlyDtm', 'DNRDtm', ...
                    'BriefOpDtm', 'HasBriefOpNote', ...
                    'O2_Delivery', 'ABO_Interpretation', 'ABO_Rh_Confirmation', 'ABO_RH_Interpretation', 'IsDc', 'los'};
treatments = {'anti_infectives_cephalosporins' 'anti_infectives_glycopeptide_antibiotics' 'anti_infectives_macrolide_derivatives' 'anti_infectives_penicillins' ...
    'cardiovascular_agents_diuretics' 'central_nervous_system_agents_analgesics' 'central_nervous_system_agents_antiemetic_antivertigo_agents' ...
    'central_nervous_system_agents_anxiolytics_sedatives_and_hypnoti' 'central_nervous_system_agents_general_anesthetics' 'coagulation_modifiers_anticoagulants' ...
    'hormones_hormone_modifiers_adrenal_cortical_steroids' 'metabolic_agents_antidiabetic_agents' 'nutritional_products_intravenous_nutritional_products' ...
    'nutritional_products_minerals_and_electrolytes' 'respiratory_agents_bronchodilators'};
excludeVars = [excludeVars treatments {'DNR_CMO_Dtm', 'AdmitToDNR_hrs', 'HasDNR_CMO', 'EDtoAdmit_hrs', 'Vent_Outcome'}];

% load imputated datasets
mi = MultipleImputations(ds, impFiles);

% concatenate the impuatations
dsImpAll = [];
for iimp = 1:length(mi.imp)
    % make language two categories
    if ~ismember('Language', mi.imp{iimp}.Properties.VariableNames) && ismember('Language', ds.Properties.VariableNames)
        mi.imp{iimp} = addvars(mi.imp{iimp}, categorical(ds.Language == 'ENGLISH', 0:1, {'Not English', 'English'}), 'NewVariableNames', {'Language'});
    end
    dsImpAll = [dsImpAll; mi.imp{iimp}];  %#ok
end

keepInds = ~ismember(dsImpAll.Properties.VariableNames, excludeVars);

% expand the categorical variables
[X, dummyStruct] = dummytable(dsImpAll(:, keepInds), true);

dev1stop = datetime(2020, 4, 23);  % development set cutoff
followup = days(28);               % final date of follow-up
lastDay = datetime(2022, 5, 1) - followup;  % admitDtm cutoff
hospital = {'LIJ'};  % hospital to hold out for retrospective validation
includeNotYetDischarged = true;  % discharge not necessary for 7 or 28-day outcomes
admitBefore = true;  % cutoffs are with respect to admissions

los = max(1, hours(ds.DischargeDtm - ds.AdmitDtm));
maxTime = max(max(ds.DischargeDtm), max(ds.AdmitDtm));
if ~ismember('IsDc', ds.Properties.VariableNames)
    ds = addvars(ds, ~isnat(ds.DischargeDtm), 'NewVariableNames', {'IsDc'});
end
los(~ds.IsDc) = hours(maxTime - ds.AdmitDtm(~ds.IsDc));  % IsDc is true if the patient was discharged (alive or dead) and 0 if the patient is still in the hospital

r1 = struct('ds', ds, 'los', los, 'measurementTime', 0, 'dateshift', days(0));
for ihorizon = 1:length(horizons)
    horizon = horizons(ihorizon);

    % get the indicies for positive and negative outcomes from each cohort
    % exclusion criteria are introduced here as well
    [indsSdev, indsDdev, indsSretro, indsDretro, indsSpro, indsDpro] = training_and_validation_set_2(r1, ...
        includeNotYetDischarged, horizon, hospital, dev1stop, lastDay, Inf, [], admitBefore, 'Expired_Outcome', true, false);
    % indsSDdev = indsSdev | indsDdev;
    % indsSDretro = indsSretro | indsDretro;
    indsSDpro = indsSpro | indsDpro;
    assert(nnz(r1.ds.AdmitDtm(indsSDpro) < dev1stop) == 0);

    % store the outcome
    y = indsSdev | indsSretro | indsSpro;    
    y2 = NaN(size(y));
    y2(y) = 1;
    noty = indsDdev | indsDretro | indsDpro;
    y2(noty) = 0;
    admitDtm = r1.ds.AdmitDtm;
    cvguid = r1.ds.Index;
    finalHospital = r1.ds.FinalHospital;
    inds = ~isnan(y2);
    y2 = y2(inds);

    % store the predictors
    m = length(impFiles);
    temp = reshape(1:size(X, 1), length(y), m);
    impInds = arrayfun(@(x) temp(:, x), 1:m, 'UniformOutput', false);
    Xmi = cellfun(@(x) X(x(inds), :), impInds, 'UniformOutput', false);

    % store metadata like hospital and patient index
    % X = X(inds, :);
    admitDtm = admitDtm(inds);
    cvguid = cvguid(inds);
    finalHospital = finalHospital(inds);
    predictorNames = [dummyStruct.names];

    [sv, si] = sort(admitDtm);
    % X = X(si, :);
    Xmi = cellfun(@(x) x(si, :), Xmi, 'UniformOutput', false);

    if ihorizon == 1
        refXmi = Xmi;
    else
        assert(isequal(Xmi, refXmi));
    end

    y2 = y2(si);
    eval(sprintf('outcome%d = y2;', horizon/24));
    admitDtm = sv;
    cvguid = cvguid(si);
    finalHospital = finalHospital(si);
    fprintf(1, 'completed horizon %d\n', horizon/24);
end
outcomes = arrayfun(@(x) sprintf('outcome%d', x), horizons/24, 'UniformOutput', false);

% save the output mat file
save(sprintf('X_Spring_2022.mat'), 'Xmi', outcomes{:}, 'predictorNames', 'admitDtm', 'cvguid', 'finalHospital');