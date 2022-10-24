% Recreate a simplified version sliding_window_pro_monitor_v2 
% Encapsulate a lot of the code within objects

% I shouldn't need to standardize these features because
% Nocos does this internally, lassoglm uses standardization by default, and XGBoost is tree-based

if ~exist('runbatch', 'var') || ~runbatch
    close all
    clear
    
    % model settings
    horizon = 28;
    modelType = 'nocos';  % {'xgboost', 'nocos', 'LR'}
end

k = 6;     % number of predictors
causal = true;
oldCausal = false;
devDate = datetime(2020, 4, 23);  % Not very interesting at datetime(2020, 4, 23)
origPredictors = {'Age' 'Blood_Urea_Nitrogen_Serum' 'Red_Cell_Distrib_Width' 'SPO2' 'Auto_Neutrophil_' 'Sodium_Serum'};
useOriginalPredictors = false;
unbiased = true;  % unbiased true retrains the Nocos model a second time after feature selection
% multipleImputations = false;
alwaysUpdate = false;
generateReport = true;

if isequal(modelType, 'xgboost')
    chosenUpdateMethod = 'update_intercept';
else
    chosenUpdateMethod = 'logsitic_recalibration';
end

% update window settings
minWindowLengths = repmat(2000, [1 5]);
stride = min(500, minWindowLengths(1));
eavgThresh = 0.03;

% plot settings
myColors = [0 0.4470 0.7410; ...
    0.8500    0.3250    0.0980; ...
    0.9290    0.6940    0.1250; ...
    0.4940    0.1840    0.5560; ...
    0.4660    0.6740    0.1880; ...
    0.3010    0.7450    0.9330; ...
    0.6350    0.0780    0.1840];
myColors = myColors([2 3 4 1 5 6 7], :);
updateMethods = [{'no updates'} BinaryClassifier.updateMethods];  % TODO verify reestimation methods
lineWidths = [6 4 2 2 2 2 2];

% If setting to true, also draw Lowess smoothed plot in Performance > p_vs_p
plotVariantGenderRaceEth = false;  % for variant, gender, race/ethnicity sensitivity analysis

saveMdl = false;  % For Lucas

%% Load the data and train the initial model
% Load dataset
if ~exist('X', 'var') || ~exist('y2', 'var')        
    load('X_Spring_2022.mat', 'Xmi', 'outcome7', 'outcome28', 'admitDtm', 'cvguid', 'finalHospital');
    if horizon == 7
        y2 = outcome7;
    elseif horizon == 28
        y2 = outcome28;
    end
    % admitDtm is corrected, and X and y2 are sorted and indexed
    if length(y2) == length(Xmi) * size(Xmi{1}, 1)
        y2 = reshape(y2, 5, [])';
        y2 = y2(:, 1);
    end
    assert(issorted(admitDtm));

    for ii = 1:length(Xmi)
        if ~ismember('eGFR', Xmi{ii}.Properties.VariableNames)
            eGFR = zeros(size(Xmi{ii}, 1), 1);
            eGFR(logical(Xmi{ii}.Race_Black)) = Xmi{ii}.eGFR_if_African_American(logical(Xmi{ii}.Race_Black));
            eGFR(~logical(Xmi{ii}.Race_Black)) = Xmi{ii}.eGFR_if_Non_African_American(~logical(Xmi{ii}.Race_Black));
            Xmi{ii} = addvars(Xmi{ii}, eGFR, 'NewVariableNames', {'eGFR'});
            Xmi{ii}.eGFR_if_African_American = [];
            Xmi{ii}.eGFR_if_Non_African_American = [];            
        end
    end
end

numImputations = length(Xmi);

% if multipleImputations
%     numImputations = length(Xmi);
    inds = reshape(reshape(1:length(y2) * numImputations, length(y2), numImputations)', [], 1);        
    X = vertcat(Xmi{1:numImputations});
    X = X(inds, :);
    stride = stride * numImputations;
    minWindowLengths = minWindowLengths * numImputations;
    y2 = reshape(repmat(y2, [1 numImputations])', [], 1);
    admitDtm = reshape(repmat(admitDtm, [1 numImputations])', [], 1);
    cvguid = reshape(repmat(cvguid, [1 numImputations])', [], 1);
    impvec = reshape(repmat(1:numImputations, [length(cvguid) 1])', [], 1);
    finalHospital = reshape(repmat(finalHospital, [1 numImputations])', [], 1);
% else
%     numImputations = 1;
%     X = Xmi{1};
% end

devInds = admitDtm < devDate & finalHospital ~= leaveoutHospital;

% ------- variants, gender, race/ethnicity -------
if plotVariantGenderRaceEth
    variants = {'alpha', 'delta', 'omicron'};
    genders = {'male', 'female'};
    raceEths = {'White' 'Black' 'Asian' 'Hispanic or Latino' 'Other'};

    alphaInds = admitDtm > devDate & admitDtm < datetime(2021, 6, 15);
    deltaInds = admitDtm > datetime(2021, 6, 15) & admitDtm < datetime(2021, 12, 15);
    omicronInds = admitDtm > datetime(2021, 12, 15) & admitDtm < datetime(2022, 5, 1);
    varIndsMap = containers.Map(variants, {alphaInds, deltaInds, omicronInds});

    maleInds = logical(X.Gender_Male);
    femaleInds = ~maleInds;
    genderIndsMap = containers.Map(genders, {maleInds, femaleInds});

    whiteInds = logical(X.Race_White) & ~logical(X.("Ethnicity_Hispanic or Latino"));
    blackInds = logical(X.Race_Black);
    % dummytable removes the first category since it is determined by the absense of all other categories
    asianInds = ~logical(X.Race_Black) & ~logical(X.Race_White) & ~logical(X.Race_Unknown) & ~logical(X.Race_Declined) & ~logical(X.Race_Other);
    hispanicLatinoInds = logical(X.("Ethnicity_Hispanic or Latino")) & ~asianInds & ~blackInds;
    otherInds = ~asianInds & ~blackInds & ~whiteInds & ~hispanicLatinoInds;
    raceEthIndsMap = containers.Map(raceEths, {whiteInds, blackInds, asianInds, hispanicLatinoInds, otherInds});
end
% ------------------------------------------------

% instantiate the model
switch modelType
    case 'nocos'
        mdl0 = NocosBC();
        params = {{} {} {} {'Lambda', 0} {'Unbiased', unbiased, 'NumFeatures', k}};
    case 'xgboost'
        mdl0 = XGBoostBC();
        params = {{} {} {} {} {'Unbiased', unbiased, 'NumFeatures', k}};
    case 'LR'
        mdl0 = LassoGlmBC();
        updateMethods = [updateMethods(1) {'dynamicLR'} updateMethods(2:end)];
        params = {{} {} {} {} {'Lambda', 0} {'Lambda', logspace(-2, -1, 50), 'Unbiased', unbiased, 'NumFeatures', k}};
        myColors = [myColors(1, :); myColors(6, :); myColors(2:end, :)];
        lineWidths = [lineWidths(1) lineWidths(6) lineWidths(2:end)];
end

% updateMethods = updateMethods(1:end-1);

% train the initial model
Xdev = X(devInds, :);
ydev = y2(devInds);
if useOriginalPredictors 
    predInds = get_pred_inds(X.Properties.VariableNames, origPredictors)    
    mdl0.train(Xdev(:, predInds), ydev, 'NumFeatures', length(predInds));
else
    predInds = 1:size(X, 2);
    mdl0.train(Xdev(:, predInds), ydev, 'NumFeatures', k);    
    
    if unbiased
        predInds = get_pred_inds(X.Properties.VariableNames, mdl0.predictorNames);      
        mdl0.train(Xdev(:, predInds), ydev, 'NumFeatures', k);
    end
end
mdlName = sprintf('mdl0_%d_%s.mat', horizon, modelType);
mdlName2 = sprintf('mdl0_%d_%s_copy.mat', horizon, modelType);
save(mdlName, 'mdl0');

% verify the histograms
% P1 = mdl0.predict(Xdev);               % apparent validation
% P2 = mdl0.predict(X(12001:13000, :));  % prospective validation
% figure;histogram(P1, 'Normalization', 'pdf');hold on;histogram(P2, 'Normalization', 'pdf');hold off;

% initialize figures
hFigVsPatients = figure(); ax1 = subplot(2, 1, 1); hold(ax1, 'on'); title(ax1, sprintf('%s, %d-day model', modelType, horizon)); ylabel(ax1, 'AUC'); grid(ax1, 'on'); ylim(ax1, [0.65 0.85]);
ax2 = subplot(2, 1, 2); hold(ax2, 'on'); ylabel(ax2, 'ICI'); grid(ax2, 'on'); xlabel(ax2, sprintf('Patients after %s', datestr(devDate, 'mmmm dd, yyyy'))); ylim(ax2, [0 0.15]);
hFigVsDate = figure(); ax1b = subplot(2, 1, 1); hold(ax1b, 'on'); title(ax1b, sprintf('%s, %d-day model', modelType, horizon)); ylabel(ax1b, 'AUC'); grid(ax1b, 'on'); ylim(ax1b, [0.65 0.85]);
ax2b = subplot(2, 1, 2); hold(ax2b, 'on'); ylabel(ax2b, 'ICI'); grid(ax2b, 'on'); ylim(ax2b, [0 0.15]);
hFigRoc = figure(); ax3 = gca(); xlabel(ax3, '1 - Specificity'); ylabel(ax3, 'Sensitivity'); title(ax3, 'ROC Curves'); grid('on');
hFigPR = figure(); ax4 = gca(); xlabel(ax4, 'Recall'); ylabel(ax4, 'Precision'); title(ax4, 'PR Curves'); grid('on');
hFigNB = figure(); ax5 = gca(); title(ax5, sprintf('%d-day NB', horizon));

% ------- variants, gender, race/ethnicity -------
if plotVariantGenderRaceEth
    hFigRocVar = figure(); axRocVar = gca(); hold(axRocVar, 'on');
    hFigPrVar = figure(); axPrVar = gca(); hold(axPrVar, 'on');
    hFigRocGender = figure(); axRocGender = gca(); hold(axRocGender, 'on');
    hFigPrGender = figure(); axPrGender = gca(); hold(axPrGender, 'on');
    hFigRocRaceEth = figure(); axRocRaceEth = gca(); hold(axRocRaceEth, 'on');
    hFigPrRaceEth = figure(); axPrRaceEth = gca(); hold(axPrRaceEth, 'on');
end
% ------------------------------------------------

figdir = sprintf('figs\\%dday\\%s', horizon, modelType);
if ~isfolder(figdir)
    mkdir(figdir);
end

%% Report generator
if generateReport && ~exist('slides', 'var')
    if exist('hcleanup', 'var') && isvalid(hcleanup)
        delete(hcleanup)
    end
    
    [hasreportgen, errmsg] = license('checkout', 'MATLAB_Report_Gen'); 
    if hasreportgen
        presentationFile = fullfile('figs', 'testPresentation.pptx');
        templateFile = fullfile('figs', 'nocos_figures_template_v2.potx');
        import mlreportgen.ppt.*
        slides = Presentation(presentationFile, templateFile);
        
        myLayouts = {'Six Three Three', 'Two by Three', 'Two by Three', 'Six Three', 'Two by Three', 'Two by Three', ...
            'Two by Three', 'Two by Three', 'Six Three Three', 'Six Three'};
        for ilayout = 1:length(myLayouts)
            add(slides, myLayouts{ilayout});
        end
        
        islide = 0;  % keep track of current slide
        hcleanup = onCleanup(@() close(slides));     
        
    else
        generateReport = false;
    end
end

%% Retrospective Validation
retroInds = admitDtm < devDate & finalHospital == leaveoutHospital;
Xretro = X(retroInds, :);
yretro = y2(retroInds);

pretro = mdl0.predict(Xretro);
if ~isreal(pretro)
    assert(max(abs(imag(pretro))) < 1e-8)
    pretro = real(pretro);
end
updateMethod = 'Retrospective Validation';
col = [0 0 0];

% ROC, PR, calibration plot, NB
plot_roc(ax3, pretro, yretro, col, updateMethod);
plot_pr(ax4, pretro, yretro, col, updateMethod); ylim(ax4, [0.75 1]);

perf = Performance(pretro, yretro, pretro);
perf.p_vs_p(0.1, false, '', horizon);
title(strrep(updateMethod, '_', ' '));
hCalRetro = gcf();
saveas(hCalRetro, sprintf('%s\\cal_%s_%d_%s.fig', figdir, modelType, horizon, 'retro'));

perf.net_benefit(hFigNB, updateMethod, col, true);

if length(leaveoutHospitals) > 1
    dnroc = get(findobj(ax3, 'Type', 'Line'), 'DisplayName');
    dnpr = get(findobj(ax4, 'Type', 'Line'), 'DisplayName');
    strCal = get(findobj(hCalRetro, 'Type', 'Text'), 'String');

    temp = strsplit(dnroc, '='); aurocvals(iloh, :, ihorizon, imodeltype) = str2num(temp{2});
    temp = strsplit(dnpr, '='); auprvals(iloh, :, ihorizon, imodeltype) = str2num(temp{2});
    temp = strsplit(strCal, '='); icivals(iloh, :, ihorizon, imodeltype) = str2num(temp{2});
    return
end

%% Dynamic updating
% window lengths
[nFlr, nIo] = get_min_sample_size(y2, 1);
nSame = get_min_sample_size(y2, k);
nExtend = get_min_sample_size(y2, 14);
% windowlenghts = max(ceil([500 nIo nFlr nSame nExtend]), minWindowLengths);
windowlenghts = minWindowLengths;

if isequal(updateMethods{2}, 'dynamicLR')
    windowlenghts = [windowlenghts(1) windowlenghts(1) windowlenghts(2:end)];
end

% window positions
ind0 = find(admitDtm >= devDate, 1, 'first') + stride;
myInds = ind0:stride:size(X, 1);

xvals2 = reshape(repmat(reshape(admitDtm(myInds), 1, []), [2 1]), [], 1);    
xvals = reshape(repmat(1:length(myInds), [2 1]), [], 1);
xvals = stride * (xvals - 1);

perf = arrayfun(@(x) Performance(), 1:length(updateMethods));
% for iupdateMethod = find(ismember(updateMethods, {'no updates', chosenUpdateMethod}))
for iupdateMethod = 1:length(updateMethods)
    updateMethod = updateMethods{iupdateMethod};
    windowlen = windowlenghts(iupdateMethod);
        
    delete(mdl0);
    load(mdlName, 'mdl0');
    mdl = mdl0;

    waitingToUpdate = false;
    oldCausal = isequal(updateMethod, 'dynamicLR');
    
    hmsg = MessageUpdater();
    [aucs, eavg] = deal(zeros(length(myInds), 2));
    [allPs, allys, allcvguidimps] = deal([]);
    w = [];
    predNames = [];
    keepInds = [];
    for ii = 1:length(myInds)
        myInd = myInds(ii);
        
        % get current data
        inds = max(1, myInd - windowlen + 1):myInd;

        % keep track of patient and imputation so that I can run DeLong
        mycvguids = cvguid(inds);
        myimps = impvec(inds);
        
        Xc = X(inds, :);
        yc = y2(inds);
        if waitingToUpdate && ismember(followupInd, inds)
            % time to update            
            indsBeforeUpdate = inds(inds < followupInd);
            indsPastUpdate = inds(inds >= followupInd); 
            Xc1 = X(indsBeforeUpdate, :);
            Xc2 = X(indsPastUpdate, :);            
            
            % use both models on all data to show discontinuities
            P1 = mdl.predict(Xc);
            P2 = mdl2.predict(Xc);

            % show the discontinuity for figure 3
            inds_ = isfinite(P1);  % can occur if point is very far from mean and reuslts in 0/0
            inds_2 = isfinite(P2);
            aucs(ii, 1) = auc_metric(P1(inds_), yc(inds_));
            eavg(ii, 1) = eavg_metric(P1(inds_), yc(inds_));
            aucs(ii, 2) = auc_metric(P2(inds_2), yc(inds_2));
            eavg(ii, 2) = eavg_metric(P2(inds_2), yc(inds_2));

            % split into two parts for overall results
            P1 = [mdl.predict(Xc1); mdl2.predict(Xc2)];
            inds_ = isfinite(P1);

            mdl = mdl2;
            clear mdl2
            waitingToUpdate = false;
        else
            % evaluate the current model
            % use the full window length so I can better estimate AUC and ICI
            P1 = mdl.predict(Xc);

            inds_ = isfinite(P1);  % can occur if point is very far from mean and reuslts in 0/0
            aucs(ii, 1) = auc_metric(P1(inds_), yc(inds_));
            eavg(ii, 1) = eavg_metric(P1(inds_), yc(inds_));
            aucs(ii, 2) = aucs(ii, 1);
            eavg(ii, 2) = eavg(ii, 1);            
        end
        % always update
        P1_ = P1(inds_);
        yc2 = yc(inds_);
        mycvguids = mycvguids(inds_);
        myimps = myimps(inds_);

        % if ~oldCausal
        allPs = [allPs; P1_(max(1, end - stride + 1):end)];
        allcvguidimps = [allcvguidimps; ...
            mycvguids(max(1, end - stride + 1):end), myimps(max(1, end - stride + 1):end)];

        allys = [allys; yc2(max(1, end - stride + 1):end)]; 

        temp = inds(inds_);
        keepInds = [keepInds, temp(max(1, end - stride + 1):end)];
        
        % decide to update based on most up-to-date model
        if ~isequal(updateMethod, 'no updates') && ...
                (eavg(ii, 2) > eavgThresh || alwaysUpdate || isequal(updateMethod, 'dynamicLR')) && ...
                ~waitingToUpdate
            % update the model
            if causal
                if oldCausal
                    % use data from the past for the update to allow the follow-up period to be met
                    % TODO this is not exactly the same as developing a new model
                    % based on the current data and not applying that model for
                    % the follow-up period.
                    followupInd = find(admitDtm < admitDtm(inds(end)) - days(horizon), 1, 'last');
                    assert(mod(followupInd, numImputations) == 0);
                    indsFollowup =  max(1, followupInd - windowlen + 1):followupInd;
                    Xcausal = X(indsFollowup, :);
                    ycausal = y2(indsFollowup);
                    mdl.(updateMethod)(Xcausal, ycausal, params{iupdateMethod}{:});

                    % aucs(ii, 2), eavg(ii, 2), allPs
                else
                    waitingToUpdate = true;
                    followupInd = find(admitDtm >= admitDtm(inds(end)) + days(horizon), 1, 'first');
                    if isempty(followupInd)
                        followupInd = length(admitDtm);
                    end

                    % It shouldn't matter what I update from, right?
                    % load(mdlName, 'mdl0');
                    % mdl2 = mdl0;
                    
                    % Make a copy of mdl by saving and then reloading
                    save(mdlName2, 'mdl');
                    tempMdl = load(mdlName2, 'mdl');
                    mdl2 = tempMdl.mdl;
                    
                    mdl2.(updateMethod)(Xc, yc, params{iupdateMethod}{:});
                end
            else
                % update using the current data
                mdl.(updateMethod)(Xc, yc, params{iupdateMethod}{:});
            end
        end        
        
        % track model parameters vs time
        w = [w; {reshape(mdl.weights(), 1, [])}];
        predNames = [predNames; {mdl.predictorNames}];
        
        hmsg.update_message(sprintf('%d of %d\n', ii, length(myInds)));
    end
    delete(hmsg);
    
    if ~isreal(allPs)
        assert(max(abs(imag(allPs))) < 1e-4);
        allPs = real(allPs);
    end

    allPs = min(1-1e-5, allPs);
    
    plot_coeffs;
    
    % plot the results
    plot(ax1, xvals, reshape(aucs', [], 1), '-', 'Color', myColors(iupdateMethod, :), 'LineWidth', lineWidths(iupdateMethod));
    plot(ax2, xvals, reshape(eavg', [], 1), '-', 'Color', myColors(iupdateMethod, :), 'LineWidth', lineWidths(iupdateMethod));
    plot(ax1b, xvals2, reshape(aucs', [], 1), '-', 'Color', myColors(iupdateMethod, :), 'LineWidth', lineWidths(iupdateMethod));
    plot(ax2b, xvals2, reshape(eavg', [], 1), '-', 'Color', myColors(iupdateMethod, :), 'LineWidth', lineWidths(iupdateMethod));
        
    % ROC, PR, calibration plot, NB
    plot_roc(ax3, allPs, allys, myColors(iupdateMethod, :), updateMethod);
    plot_pr(ax4, allPs, allys, myColors(iupdateMethod, :), updateMethod); ylim(ax4, [0.75 1]);

    perf(iupdateMethod).init(allPs, allys, allPs, true, allcvguidimps);
    perf(iupdateMethod).p_vs_p(0.1, false, '', horizon, numImputations);        
    title(strrep(updateMethod, '_', ' '));
    hCal = gcf();
    saveas(hCal, sprintf('%s\\cal_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));

    % TODO ROC/PR/Cal for each variant
    % ------- variants, gender, race/ethnicity -------
    if plotVariantGenderRaceEth        
        % each model type can potentially have a different preferred updating method
        % only plot the results for nocos
        if ismember(modelType, {'nocos', 'LR'}) && ismember(updateMethod, {'no updates', 'logistic_recalibration'}) || ...
            ismember(modelType, {'xgboost'}) && ismember(updateMethod, {'no updates', 'update_intercept'})
            eavgs = [];
            [meanVarNauroc, meanVarNaupr, meanVarNici] = deal(zeros(length(variants), 3));
            for ivar = 1:length(variants)
                variant = variants{ivar};
                varInds = varIndsMap(variant);
                varInds_ = varInds(keepInds);
                [meanVarNauroc(ivar, :)] = plot_roc(axRocVar, allPs(varInds_), allys(varInds_), myColors(ivar, :), variant);
                [meanVarNaupr(ivar, :)] = plot_pr(axPrVar, allPs(varInds_), allys(varInds_), myColors(ivar, :), variant); ylim(axPrVar, [0.75 1]);

                perf_ = Performance();
                perf_.init(allPs(varInds_), allys(varInds_), allPs(varInds_), true, allcvguidimps(varInds_));
                [meanVarNici(ivar, :), ~, ~, Eavg_] = perf_.p_vs_p(0.1, false, '', horizon, numImputations);
                title(variant);
                hFigVarCal = gcf();
                saveas(hFigVarCal, sprintf('%s\\cal_%s_%d_%s_%s.fig', figdir, modelType, horizon, variant, updateMethod));

                % I think these are normally distributed
                eavgs = [eavgs; Eavg_, ivar + zeros(length(Eavg_), 1)];
            end
            saveas(hFigRocVar, sprintf('%s\\roc_variants_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));
            saveas(hFigPrVar, sprintf('%s\\pr_variants_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));            
            
            % groupVariants = onehotdecode([alphaInds deltaInds omicronInds], ["alpha" "delta" "omicron"], 1);            
            [pVariantIci, tblVariantIci, statsVariantIci] = anova1(eavgs(:, 1), categorical(eavgs(:, 2), 1:length(variants), variants));
            title(gca(), sprintf('%s %s %d %f', modelType, updateMethod, horizon, pVariantIci));
            saveas(gca(), sprintf('%s\\anova_boxplots_variants_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));
            % [hVariantAuroc2, pVariantAuroc2] = anova1ss(meanVarNauroc(:, 1), meanVarNauroc(:, 2), meanVarNauroc(:, 3));
            % [hVariantAupr2, pVariantAupr2] = anova1ss(meanVarNaupr(:, 1), meanVarNaupr(:, 2), meanVarNaupr(:, 3));
            % [hVariantIci2, pVariantIci2] = anova1ss(meanVarNici(:, 1), meanVarNici(:, 2), meanVarNici(:, 3));

            % TODO if I did this for AUROC and logit(AUPR) what
            % would N be? Would it be the number of samples in the group?
            % All samples are required to compute the metric. But intrinsically my calculations use that n.
            % The Hanley and MacNeil SE scales as sqrt(n) so I assume use the number of samples

            % I have the 95% confidence intervals for the AUROC, AUPR, and ICI
            % ICI was the only one I did bootstrapping for. Do I have to
            % ignore my analytically computed confidence intervals and
            % perform bootstrapping for AUROC and AUPR too to run ANOVA?            
            
            % invlogit = @(x) exp(x) ./ (1 + exp(x));
            % logit = @(x) log(x ./ (1 - x));

            % I could do 3-way ANOVA to compare White Alpha Males
            % multcompare.m is pairwise.

            % One-way ANOVA from summary data
            % https://statpages.info/anova1sm.html
            % https://www.mathworks.com/matlabcentral/fileexchange/41036-n-way-anova-from-summary-statistics
            % maybe I can implement it from this https://www.originlab.com/doc/Origin-Help/OneWayANOVA-Algorithm

            % here's a paper that compares AUCs, but it doesn't make the comparison between races, only across models
            % https://www.ahajournals.org/doi/10.1161/JAHA.118.010471

            eavgs = [];
            for igender = 1:length(genders)
                gender = genders{igender};
                genderInds = genderIndsMap(gender);
                genderInds_ = genderInds(keepInds);
                plot_roc(axRocGender, allPs(genderInds_), allys(genderInds_), myColors(igender, :), gender);
                plot_pr(axPrGender, allPs(genderInds_), allys(genderInds_), myColors(igender, :), gender); ylim(axPrGender, [0.75 1]);

                perf_ = Performance();
                perf_.init(allPs(genderInds_), allys(genderInds_), allPs(genderInds_), true, allcvguidimps(genderInds_));
                [~, ~, ~, Eavg_] = perf_.p_vs_p(0.1, false, '', horizon, numImputations);
                title(gender);
                hFigGenderCal = gcf();
                saveas(hFigGenderCal, sprintf('%s\\cal_%s_%d_%s_%s.fig', figdir, modelType, horizon, gender, updateMethod));

                eavgs = [eavgs; Eavg_, igender + zeros(length(Eavg_), 1)];
            end
            saveas(hFigRocGender, sprintf('%s\\roc_gender_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));
            saveas(hFigPrGender, sprintf('%s\\pr_gender_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));
            % groupGenders = onehotdecode([maleInds femaleInds], ["male" "female"], 1);
            % [pGender, tblGender, statsGender] = anova1(allPs(keepInds) - allys(keepInds), groupGenders(keepInds));
            [pGenderIci, tblGenderIci, statsGenderIci] = anova1(eavgs(:, 1), categorical(eavgs(:, 2), 1:length(genders), genders));
            title(gca(), sprintf('%s %s %d %f', modelType, updateMethod, horizon, pGenderIci));
            saveas(gca(), sprintf('%s\\anova_boxplots_gender_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));
            

            eavgs = [];
            for ire = 1:length(raceEths)
                raceEth = raceEths{ire};
                raceEthInds = raceEthIndsMap(raceEth);
                raceEthInds_ = raceEthInds(keepInds);
                plot_roc(axRocRaceEth, allPs(raceEthInds_), allys(raceEthInds_), myColors(ire, :), raceEth);
                plot_pr(axPrRaceEth, allPs(raceEthInds_), allys(raceEthInds_), myColors(ire, :), raceEth); ylim(axPrRaceEth, [0.75 1]);

                perf_ = Performance();
                perf_.init(allPs(raceEthInds_), allys(raceEthInds_), allPs(raceEthInds_), true, allcvguidimps(raceEthInds_));
                [~, ~, ~, Eavg_] = perf_.p_vs_p(0.1, false, '', horizon, numImputations);
                title(raceEth);
                hFigRaceEthCal = gcf();
                saveas(hFigRaceEthCal, sprintf('%s\\cal_%s_%d_%s_%s.fig', figdir, modelType, horizon, raceEth, updateMethod));

                eavgs = [eavgs; Eavg_, ire + zeros(length(Eavg_), 1)];
            end
            saveas(hFigRocRaceEth, sprintf('%s\\roc_raceEth_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));
            saveas(hFigPrRaceEth, sprintf('%s\\pr_raceEth_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));
            % groupRaceEth = onehotdecode([whiteInds blackInds asianInds hispanicLatinoInds otherInds], ["White" "Black" "Asian" "Hispanic or Latino" "Other"], 1);
            % [pRaceEth, tblRaceEth, statsRaceEth] = anova1(allPs(keepInds) - allys(keepInds), groupRaceEth(keepInds));
            [pRaceEthIci, tblRaceEthIci, statsRaceEthIci] = anova1(eavgs(:, 1), categorical(eavgs(:, 2), 1:length(raceEths), raceEths));
            title(gca(), sprintf('%s %s %d %f', modelType, updateMethod, horizon, pRaceEthIci));
            saveas(gca(), sprintf('%s\\anova_boxplots_raceEth_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));
        end        
    end
    % ------------------------------------------------
    
    perf(iupdateMethod).net_benefit(hFigNB, updateMethod, myColors(iupdateMethod, :), iupdateMethod == 1); 
    drawnow();
    
    % chosen update method slides
    if generateReport
        report1;
    end

    if saveMdl && isequal(updateMethod, chosenUpdateMethod)
        break
    end
end

if saveMdl
    mdlName3 = sprintf('mdl_%d_%s_final.mat', horizon, modelType);
    myMFile = sprintf('mdl_%d_%s_final.m', horizon, modelType);
    save(mdlName3, 'mdl');
    % what method writes the parameters?
    mdl.mdl.calCorrection = [mdl.alphaNew mdl.betaOverall];
    mdl.mdl.generate_eval_function(myMFile);
    return
end

pdl = NaN(length(updateMethods), length(updateMethods));
for ium = 1:length(updateMethods)
    for ium2 = setdiff(1:length(updateMethods), ium)
        try
            % pdl(ium, ium2) = perf(ium).deLong(perf(ium2));
        catch
            % This can happen if the in NaNs cause me to remove indicies
            % and the results are no longer pairwise
            % TODO I would need to get the ClientVisitGUIDs and intersect them
        end
    end
end
% save(sprintf('%s\\pDeLong_%s_%d.mat', figdir, modelType, horizon), 'pdl');

% add pdl to legend
% um2 = arrayfun(@(x) sprintf('%s (p = %1.3f)', strrep(updateMethods{x+1}, '_', ' '), pdl(x)), ...
%     1:length(pdl), 'UniformOutput', false);
legend(ax1, strrep(updateMethods, '_', ' '));
legend(ax1b, strrep(updateMethods, '_', ' '));

legend(ax3, 'Location', 'southeast');
legend(ax4, 'Location', 'southwest');

xlim(ax1, xvals([1 end])); ylim(ax1, [0.7 0.85]);
xlim(ax2, xvals([1 end]));
hold(ax2, 'on'); plot(ax2, xvals([1 end]), repmat(eavgThresh, [2 1]), '--k'); hold(ax2, 'off');

xlim(ax1b, xvals2([1 end])); ylim(ax1b, [0.7 0.85]);
xlim(ax2b, xvals2([1 end]));
hold(ax2b, 'on'); plot(ax2b, xvals2([1 end]), repmat(eavgThresh, [2 1]), '--k'); hold(ax2b, 'off');

saveas(hFigVsPatients, sprintf('%s\\auc_ici_vs_patients_%s_%d.fig', figdir, modelType, horizon));
saveas(hFigVsDate, sprintf('%s\\auc_ici_vs_date_%s_%d.fig', figdir, modelType, horizon));
saveas(hFigRoc, sprintf('%s\\roc_%s_%d.fig', figdir, modelType, horizon));
saveas(hFigPR, sprintf('%s\\pr_%s_%d.fig', figdir, modelType, horizon));
saveas(hFigNB, sprintf('%s\\nb_%s_%d.fig', figdir, modelType, horizon));

% all method slides
if generateReport
    report2;
end