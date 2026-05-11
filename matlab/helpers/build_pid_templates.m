function sources = build_pid_templates(cfg, csDecoded, csOnlyDecoded, doorDecodedCS, doorDecodedNoCS)
%BUILD_PID_TEMPLATES Build X1 and X2 source matrices for PID.
%
% Each template is one time-normalized trial. The caller repeats columns for
% each condition/trial before passing to quickPID.

fixLen = cfg.pid.fixLen;
csOn = cfg.timing.baselineLength;
doorOn = cfg.pid.fixedPreOpenBins;

csDecoded = pad_or_trim(csDecoded(:), fixLen);
csOnlyDecoded = pad_or_trim(csOnlyDecoded(:), fixLen);
doorDecodedCS = pad_or_trim(doorDecodedCS(:), fixLen);
doorDecodedNoCS = pad_or_trim(doorDecodedNoCS(:), fixLen);

sources.X1_CS = zeros(fixLen, 1);
sources.X1_CS(csOn:fixLen) = csDecoded(csOn:fixLen);
sources.X2_CS = zeros(fixLen, 1);
sources.X2_CS(doorOn:fixLen) = doorDecodedCS(doorOn:fixLen);

sources.X1_openOnly = zeros(fixLen, 1);
sources.X2_openOnly = zeros(fixLen, 1);
sources.X2_openOnly(doorOn:fixLen) = doorDecodedNoCS(doorOn:fixLen);

sources.X1_csOnly = zeros(fixLen, 1);
sources.X1_csOnly(csOn:fixLen) = csOnlyDecoded(csOn:fixLen);
sources.X2_csOnly = zeros(fixLen, 1);

sources.X1_none = zeros(fixLen, 1);
sources.X2_none = zeros(fixLen, 1);
end

function y = pad_or_trim(x, n)
y = zeros(n, 1);
copyN = min(n, numel(x));
y(1:copyN) = x(1:copyN);
end
