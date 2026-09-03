function msbtd_probe(method_root)
addpath(method_root);
s = struct('nim', uint8(round(255 * rand(64, 64))));
for iteration = 1:30
    try
        output = Sparsity_based_SDOCT_Denoising(s);
        disp('SUCCESS');
        disp(size(output));
        break;
    catch exception
        disp(['ID=' exception.identifier]);
        disp(exception.message);
        if strcmp(exception.identifier, 'MATLAB:nonExistentField')
            token = regexp(exception.message, '"([^"]+)"', 'tokens', 'once');
            if isempty(token)
                break;
            end
            field = token{1};
            disp(['ADDING=' field]);
            s.(field) = 1;
        else
            break;
        end
    end
end
disp(fieldnames(s));
end
