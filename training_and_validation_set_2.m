function [inds1, inds2, inds1b, inds2b, inds1c, inds2c] = training_and_validation_set_2(r1, ...
    includeNotYetDischarged, horizon, hospitalsToLeaveOut, dateCutoff, lastDay, losWindow, ...
    lessThanDateCutoff, admitBefore, outcome, excludeVentBeforeAdmission, allDNR)

if ~exist('excludeVentBeforeAdmission', 'var') || isempty(excludeVentBeforeAdmission)    
    excludeVentBeforeAdmission = true;
end
if ~exist('allDNR', 'var') || isempty(allDNR)
    allDNR = false;
end

% TODO Expired_Outcome and Vented
% IsDc and los

if ~exist('includeNotYetDischarged', 'var') || isempty(includeNotYetDischarged)
    includeNotYetDischarged = true;
end
if ~exist('horizon', 'var') || isempty(horizon)
    horizon = Inf;
end
if ~exist('hospitalsToLeaveOut', 'var') || isempty(hospitalsToLeaveOut)
    hospitalsToLeaveOut = 'Long Island Jewish Hospital';
end
if ~exist('dateCutoff', 'var') || isempty(dateCutoff)
    dateCutoff = datetime(2020, 4, 23);
end
if ~exist('losWindow', 'var') || isempty(losWindow)
    losWindow = Inf;
end
if ~exist('lessThanDateCutoff', 'var') || isempty(lessThanDateCutoff)
    lessThanDateCutoff = true;
end

if isequal(outcome, 'Expired_Outcome')
    outcomeDtm = 'DischargeDtm';
    los = r1.los;
elseif isequal(outcome, 'Vented')
    outcomeDtm = 'DischargeDtm';  % 'VentDtm';  % Don't change the training and testing sets, just the distribution of positive and negative outcomes within the sets
    los = max(1, hours(r1.ds.VentDtm - r1.ds.AdmitDtm));
    maxTime = max(max(r1.ds.DischargeDtm), max(r1.ds.AdmitDtm));
    los(~r1.ds.IsDc & ~r1.ds.Vented) = hours(maxTime - r1.ds.AdmitDtm(~r1.ds.IsDc & ~r1.ds.Vented));
end

hospitals1 = setdiff(categories(r1.ds.FinalHospital), hospitalsToLeaveOut);

dischargedAlive = r1.ds.IsDc & r1.ds.(outcome) == 0;  % this is correct for either outcome
inHospitalPastSevenDays = los - r1.measurementTime * 24 >= horizon;
stillInHospital = includeNotYetDischarged & ~r1.ds.IsDc & r1.ds.(outcome) == 0;


diedAfterSevenDays = inHospitalPastSevenDays & r1.ds.(outcome) == 1;  % obj.ds.IsDc == 1 &

aliveAtSevenDays = inHospitalPastSevenDays & stillInHospital | diedAfterSevenDays;
sevenDaySurvival = dischargedAlive | aliveAtSevenDays;
notTransferred = r1.ds.OutsideTransfer ~= 1;
notAnExcludedHospital = ismember(r1.ds.FinalHospital, hospitals1);
if ismember('DNRDtm', r1.ds.Properties.VariableNames)    
    % The top line is wrong, but the bottom line might need a new model to obtain good results
    notDNR = isnat(r1.ds.DeceasedDtm) | isnat(r1.ds.DNRDtm) | days(r1.ds.DeceasedDtm - r1.ds.DNRDtm) <= 5;  % | hours(obj.ds.DeceasedDtm - obj.ds.AdmitDtm) > horizon;
    % notDNR = isnat(obj.ds.DeceasedDtm) | isnat(obj.ds.DNRDtm) | days(obj.ds.DeceasedDtm - obj.ds.DNRDtm) > -Inf;  % | hours(obj.ds.DeceasedDtm - obj.ds.AdmitDtm) > horizon;
elseif ismember('DNR_CMO_Dtm', r1.ds.Properties.VariableNames)    
    notDNR = isnat(r1.ds.DeceasedDtm) | isnat(r1.ds.DNR_CMO_Dtm) | days(r1.ds.DeceasedDtm - r1.ds.DNR_CMO_Dtm) <= 5;  % | hours(obj.ds.DeceasedDtm - obj.ds.AdmitDtm) > horizon;    
    % notDNR = isnat(obj.ds.DeceasedDtm) | isnat(obj.ds.DNR_CMO_Dtm) | days(obj.ds.DeceasedDtm - obj.ds.DNR_CMO_Dtm) > -Inf;  % | hours(obj.ds.DeceasedDtm - obj.ds.AdmitDtm) > horizon;    
else
    error('no DNR field');
end

if allDNR
    notDNR = true(size(notDNR));
end

if lessThanDateCutoff
    dischargedBeforeTrainingEndDate = r1.ds.(outcomeDtm) - r1.dateshift < dateCutoff;
else
    % Use Admit here
    dischargedBeforeTrainingEndDate = r1.ds.AdmitDtm - r1.dateshift > dateCutoff;
end
admittedAtLeastAWeekBeforeTrainingEndDate = r1.ds.AdmitDtm - r1.dateshift < dateCutoff - hours(horizon);
admittedBeforeTrainingEndDate = r1.ds.AdmitDtm - r1.dateshift < dateCutoff;
admittedBeforeValidationEndDate = r1.ds.AdmitDtm - r1.dateshift < lastDay;
dischargedAfterTrainingEndDate = r1.ds.AdmitDtm - r1.dateshift >= dateCutoff;
dischargeOrAdmittedBeforeTrainingEndDate = dischargedBeforeTrainingEndDate | isnat(r1.ds.(outcomeDtm)) & admittedBeforeTrainingEndDate;
losInRange = los < (r1.measurementTime + losWindow) * 24 & los > r1.measurementTime * 24;
notOnVentBeforeAdmission = isnan(r1.ds.AdmitToVent_hrs) | r1.ds.AdmitToVent_hrs >= 0;

% TODO fix
if excludeVentBeforeAdmission
    notExclusion = notTransferred & notAnExcludedHospital & notOnVentBeforeAdmission & notDNR;
else
    notExclusion = notTransferred & notAnExcludedHospital & notDNR;
end

if excludeVentBeforeAdmission
    notExclusionPro = notTransferred & notOnVentBeforeAdmission & notDNR;
else
    notExclusionPro = notTransferred & notDNR;
end

died = r1.ds.(outcome) == 1;  % obj.ds.IsDc & 
diedBeforeSevenDays = died & los - r1.measurementTime * 24 < horizon;

stillInHospitalButLessThanSevenDays = r1.ds.IsDc ~= 1 & ~inHospitalPastSevenDays;

%% Development Set
if admitBefore
%     inds1 = sevenDaySurvival & notExclusion & (dischargedBeforeTrainingEndDate | admittedAtLeastAWeekBeforeTrainingEndDate) & losInRange;
%     inds2 = diedBeforeSevenDays & notExclusion & (dischargedBeforeTrainingEndDate | admittedAtLeastAWeekBeforeTrainingEndDate) & losInRange;
    inds1 = sevenDaySurvival & notExclusion & admittedBeforeTrainingEndDate & losInRange;
    inds2 = diedBeforeSevenDays & notExclusion & admittedBeforeTrainingEndDate & losInRange;
    
    inds1WithAllHospitals = sevenDaySurvival & notExclusionPro & admittedBeforeTrainingEndDate & losInRange;
    inds2WithAllHospitals = diedBeforeSevenDays & notExclusionPro & admittedBeforeTrainingEndDate & losInRange;
else
    inds1 = sevenDaySurvival & notExclusion & dischargedBeforeTrainingEndDate & losInRange;
    inds2 = diedBeforeSevenDays & notExclusion & dischargedBeforeTrainingEndDate & losInRange;
    
    inds1WithAllHospitals = sevenDaySurvival & notExclusionPro & dischargedBeforeTrainingEndDate & losInRange;
    inds2WithAllHospitals = diedBeforeSevenDays & notExclusionPro & dischargedBeforeTrainingEndDate & losInRange;
end

%% Prospective Validation
inds1c = sevenDaySurvival & notExclusionPro & losInRange & ~inds1WithAllHospitals & admittedBeforeValidationEndDate;
inds2c = diedBeforeSevenDays & notExclusionPro & losInRange & ~inds2WithAllHospitals & admittedBeforeValidationEndDate;

r1.hospitalsToLeaveOut = hospitalsToLeaveOut;
r1.dateCutoff = dateCutoff;

if excludeVentBeforeAdmission
    if admitBefore
        inds1b = sevenDaySurvival & ismember(r1.ds.FinalHospital, r1.hospitalsToLeaveOut) & notTransferred & notOnVentBeforeAdmission & notDNR & admittedBeforeTrainingEndDate & losInRange;
        inds2b = diedBeforeSevenDays & ismember(r1.ds.FinalHospital, r1.hospitalsToLeaveOut) & notTransferred & notOnVentBeforeAdmission & notDNR & admittedBeforeTrainingEndDate & losInRange;
    else
        inds1b = sevenDaySurvival & ismember(r1.ds.FinalHospital, r1.hospitalsToLeaveOut) & notTransferred & notOnVentBeforeAdmission & notDNR & dischargedBeforeTrainingEndDate & losInRange;
        inds2b = diedBeforeSevenDays & ismember(r1.ds.FinalHospital, r1.hospitalsToLeaveOut) & notTransferred & notOnVentBeforeAdmission & notDNR & dischargedBeforeTrainingEndDate & losInRange;
    end
else
    if admitBefore
        inds1b = sevenDaySurvival & ismember(r1.ds.FinalHospital, r1.hospitalsToLeaveOut) & notTransferred & notDNR & admittedBeforeTrainingEndDate & losInRange;
        inds2b = diedBeforeSevenDays & ismember(r1.ds.FinalHospital, r1.hospitalsToLeaveOut) & notTransferred & notDNR & admittedBeforeTrainingEndDate & losInRange;
    else
        inds1b = sevenDaySurvival & ismember(r1.ds.FinalHospital, r1.hospitalsToLeaveOut) & notTransferred & notDNR & dischargedBeforeTrainingEndDate & losInRange;
        inds2b = diedBeforeSevenDays & ismember(r1.ds.FinalHospital, r1.hospitalsToLeaveOut) & notTransferred & notDNR & dischargedBeforeTrainingEndDate & losInRange;
    end
end

% inds1 = sevenDaySurvival & notExclusion & dischargeOrAdmittedBeforeTrainingEndDate & losInRange;
% inds2 = diedBeforeSevenDays & notExclusion & dischargeOrAdmittedBeforeTrainingEndDate & losInRange;

% nnz(stillInHospitalButLessThanSevenDays);
% nnz(~notTransferred);
% nnz(~notOnVentBeforeAdmission);
% nnz(~dischargeOrAdmittedBeforeTrainingEndDate & ~dischargedAfterTrainingEndDate);
% 
% % survived and discharged or (still in hospital or expired) with LoS >= trainingHorizon for all hospitals except one
% inds1_ = (obj.ds.IsDc & obj.ds.(outcome) == 0 | ...
%     obj.los - obj.measurementTime * 24 >= trainingHorizon & ...
%     (includeNotYetDischarged & ~obj.ds.IsDc & obj.ds.(outcome) == 0 | obj.ds.IsDc & obj.ds.(outcome) == 1)) & ...    
%     obj.ds.OutsideTransfer ~= 1 & ismember(obj.ds.FinalHospital, hospitals1) & ...
%     (obj.ds.(outcomeDtm) - obj.dateshift < dateCutoff | ...
%     isnat(obj.ds.(outcomeDtm)) & obj.ds.AdmitDtm - obj.dateshift < dateCutoff) & ...  % TODO AdmitDtm could be > cutoff
%     obj.los < (obj.measurementTime + losWindow) * 24 & obj.los > obj.measurementTime * 24 & ...
%     (isnan(obj.ds.AdmitToVent_hrs) | obj.ds.AdmitToVent_hrs > 0);
% % expired with LoS < trainingHorizon for all hospitals except one
% inds2_ = obj.ds.IsDc & obj.ds.(outcome) == 1 & obj.ds.OutsideTransfer ~= 1 & ...
%     obj.los - obj.measurementTime * 24 < trainingHorizon & ismember(obj.ds.FinalHospital, hospitals1) & ...
%     (obj.ds.(outcomeDtm) - obj.dateshift < dateCutoff | ...
%     isnat(obj.ds.(outcomeDtm)) & obj.ds.AdmitDtm - obj.dateshift < dateCutoff) & ...
%     obj.los < (obj.measurementTime + losWindow) * 24 & obj.los > obj.measurementTime * 24 & ...
%     (isnan(obj.ds.AdmitToVent_hrs) | obj.ds.AdmitToVent_hrs > 0);
% 
% assert(isequal(inds1, inds1_) && isequal(inds2, inds2_));
% 


%% Retrospective Validation
% TODO substitute other variables
% survived and discharged or (still in hospital or expired) with LoS >= horizon for one hospital
% if excludeVentBeforeAdmission
%     inds1b = (obj.ds.IsDc & obj.ds.(outcome) == 0 | ...
%         los - obj.measurementTime * 24 >= horizon & ...
%         (includeNotYetDischarged & ~obj.ds.IsDc & obj.ds.(outcome) == 0 | obj.ds.(outcome) == 1)) & ...  % obj.ds.IsDc &
%         obj.ds.OutsideTransfer ~= 1 & ismember(obj.ds.FinalHospital, obj.hospitalsToLeaveOut) & ...
%         los < (obj.measurementTime + losWindow) * 24 & los > obj.measurementTime * 24 & ...
%         admittedBeforeValidationEndDate;  % (isnan(obj.ds.AdmitToVent_hrs) | obj.ds.AdmitToVent_hrs > 0) &   % TODO fix
%     % expired with LoS < horizon for one hospital
%     inds2b = obj.ds.(outcome) == 1 & obj.ds.OutsideTransfer ~= 1 & ...  % obj.ds.IsDc &
%         los - obj.measurementTime * 24 < horizon & ...
%         ismember(obj.ds.FinalHospital, obj.hospitalsToLeaveOut) & ...
%         los < (obj.measurementTime + losWindow) * 24 & los > obj.measurementTime * 24 & ...
%         admittedBeforeValidationEndDate;  % (isnan(obj.ds.AdmitToVent_hrs) | obj.ds.AdmitToVent_hrs > 0) &  % TODO fix
% else        
%     inds1b = (obj.ds.IsDc & obj.ds.(outcome) == 0 | ...
%         los - obj.measurementTime * 24 >= horizon & ...
%         (includeNotYetDischarged & ~obj.ds.IsDc & obj.ds.(outcome) == 0 | obj.ds.(outcome) == 1)) & ...  % obj.ds.IsDc &
%         obj.ds.OutsideTransfer ~= 1 & ismember(obj.ds.FinalHospital, obj.hospitalsToLeaveOut) & ...
%         los < (obj.measurementTime + losWindow) * 24 & los > obj.measurementTime * 24 & ...
%         (isnan(obj.ds.AdmitToVent_hrs) | obj.ds.AdmitToVent_hrs > 0) & admittedBeforeValidationEndDate;   % TODO fix
%     % expired with LoS < horizon for one hospital
%     inds2b = obj.ds.(outcome) == 1 & obj.ds.OutsideTransfer ~= 1 & ...  % obj.ds.IsDc &
%         los - obj.measurementTime * 24 < horizon & ...
%         ismember(obj.ds.FinalHospital, obj.hospitalsToLeaveOut) & ...
%         los < (obj.measurementTime + losWindow) * 24 & los > obj.measurementTime * 24 & ...
%         (isnan(obj.ds.AdmitToVent_hrs) | obj.ds.AdmitToVent_hrs > 0) & admittedBeforeValidationEndDate;  % TODO fix
% end
% 
% if admitBefore
%     inds1b = inds1b & admittedBeforeTrainingEndDate;
%     inds2b = inds2b & admittedBeforeTrainingEndDate;
% else
%     inds1b = inds1b & dischargedBeforeTrainingEndDate;
%     inds2b = inds2b & dischargedBeforeTrainingEndDate;
% end
