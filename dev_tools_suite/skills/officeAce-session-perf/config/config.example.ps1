# Copy this file to config.local.ps1 (same directory) and set your paths.
# The skill loads config.local.ps1 automatically before each run.

$Script:OfficeClawSkillConfig = @{
    # history.json 所在 sessions 根目录
    SessionsRoot = "C:\Users\liutao\.office-claw\.jiuwenclaw\service_default\agent_default\agent\sessions"
    # full*.log / full*.json 所在目录
    LogsRoot     = "C:\Users\liutao\.office-claw\.jiuwenclaw\service_default\.logs"
}
