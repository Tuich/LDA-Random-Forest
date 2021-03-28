function L = list_files(path)
%% List all files in a folder and return it as a cell array.
%%
%% Args:
%%  path: Path to list files from.
%%
%% Returns:
%%  L: Cell array with files in this folder.
%%
%% Example:
%% 	L = list_files("./data/yalefaces")
L = dir(path);
if strcmpi(L(1).name, '.')
    L = L(3:length(L));
else
    L = L(1:length(L)-2);
end
L = struct2cell(L);
L = L(1,:)';
end
