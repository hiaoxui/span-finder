{
    json: function(name, default) if std.extVar("LOGNAME")=="tuning" then std.parseJson(std.extVar(name)) else std.parseJson(default),
    str: function(name, default) if std.extVar("LOGNAME")=="tuning" then std.extVar(name) else default
}