function score = decoder_score(accuracy, smoothWindow, roundDigits)
%DECODER_SCORE Convert decoding accuracy to a 0-to-1 source-strength score.
score = max((accuracy(:) - 0.5) * 2, 0);
score(~isfinite(score)) = 0;
if smoothWindow > 1
    score = smoothdata(score, 'gaussian', smoothWindow);
end
score = round(score, roundDigits);
end
