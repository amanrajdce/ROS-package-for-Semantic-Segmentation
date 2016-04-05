function f = fullfile_ext(varargin)
% function f = fullfile_ext(varargin)
% f = sprintf('%s.%s', fullfile(varargin{1:end-1}), varargin{end});

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Saurabh Gupta
% 
% This file is part of the Utils code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

  f = sprintf('%s.%s', fullfile(varargin{1:end-1}), varargin{end});
end
