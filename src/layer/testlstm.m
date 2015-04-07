Wix = [-0.079999,-0.058954; 
        0.040897,-0.006616];
Wih = [0.005243,-0.044967;
      -0.072473,0.028618];
Wic = [0.028687;
       0.069551];

Wfx = [-0.018640,0.003107;
        0.052954,-0.074468];
Wfh = [-0.071446,0.004752;
        0.027384,-0.078768];
Wfc = [-0.018653;
       -0.069305];

Wcx = [-0.013202,0.029884;
        0.014236,0.068870];
Wch = [0.055387,0.004309;
      -0.065286,0.024627];

Wox = [-0.013440,0.032190;
        0.065651,0.041952];
Woh = [-0.038008,-0.072406;
        0.037773,-0.027483];
Woc = [0.021222;
       0.041026];
   
W = [0.078566,-0.021546,0.035626;
    -0.040474,0.077208,0.040537];

% [2x1]
inputs = [0, 1, 2, 3, 4, 5, 0; 
          0, 2, 4, 6, 8, 10, 0];

states = zeros(2,7);
outputs = zeros(2,7);      
inGate = zeros(2,7);       
foGate = zeros(2,7);       
preStates = zeros(2,7);       
ouGate = zeros(2,7);       
preOutActs = zeros(2,7);
softmaxRes = zeros(2,7);
       
sigmoid = @(x) 1 ./ (1 + exp(-x));
sigmDeriv = @(x) (1-x) .* x;
tanhDeriv = @(x) (1 - x .^ 2);

for i=2:6
    inGate(:,i) = sigmoid (Wix * inputs(:,i) + Wih * outputs(:,i-1) + Wic .* states(:,i-1));
    foGate(:,i) = sigmoid (Wfx * inputs(:,i) + Wfh * outputs(:,i-1) + Wfc .* states(:,i-1));
    preStates(:,i) = tanh(Wcx * inputs(:,i) + Wch * outputs(:,i-1));
    states(:,i) = foGate(:,i) .* states(:,i-1) + inGate(:,i) .* preStates(:,i);
    ouGate(:,i) = sigmoid (Wox * inputs(:,i) + Woh * outputs(:,i-1) + Woc .* states(:,i));
    preOutActs(:,i) = tanh(states(:,i));
    outputs(:,i) = ouGate(:,i) .* preOutActs(:,i);
    softmaxRes(:,i) = W * [outputs(:,i); 1];
    softmaxRes(:,i) = exp(softmaxRes(:,i)) / sum(exp(softmaxRes(:,i)));
end

labels = [0, 18, 16, 14, 12, 10, 0; 
           0, 9, 8, 7, 6, 5, 0];
outErrs = W' * (softmaxRes - labels);
outErrs = outErrs(1:2,:);

inErrs = zeros(2,7);
statesErrs = zeros(2,7);
inGateDelta = zeros(2,7);
foGateDelta = zeros(2,7);
ouGateDelta = zeros(2,7);
preStatesDelta = zeros(2,7);

for i=6:-1:2    
    ouGateDelta(:,i) = outErrs(:,i) .* sigmDeriv(ouGate(:,i)) .* preOutActs(:,i);
    statesErrs(:,i) = outErrs(:,i) .* ouGate(:,i) .* tanhDeriv(preOutActs(:,i)) ...
                    + statesErrs(:,i+1) .* foGate(:,i+1) ...
                    + Wic .* inGateDelta(:,i+1) ...
                    + Wfc .* foGateDelta(:,i+1) ...
                    + Woc .* ouGateDelta(:,i);
    preStatesDelta(:,i) = statesErrs(:,i) .* inGate(:,i)    .* tanhDeriv(preStates(:,i));
    foGateDelta(:,i)    = statesErrs(:,i) .* states(:,i-1)  .* sigmDeriv(foGate(:,i));
    inGateDelta(:,i)    = statesErrs(:,i) .* preStates(:,i) .* sigmDeriv(inGate(:,i));
    inErrs(:,i) = Wix' * inGateDelta(:,i)    + Wfx' * foGateDelta(:,i) ...
                + Wcx' * preStatesDelta(:,i) + Wox' * ouGateDelta(:,i);
end

