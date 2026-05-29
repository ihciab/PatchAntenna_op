using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

internal static class Program
{
    private const int PROCESS_QUERY_INFORMATION = 0x0400;
    private const int PROCESS_VM_READ = 0x0010;
    private const int PROCESS_QUERY_LIMITED_INFORMATION = 0x1000;
    private const int ProcessBasicInformation = 0;

    [StructLayout(LayoutKind.Sequential)]
    private struct UNICODE_STRING
    {
        public ushort Length;
        public ushort MaximumLength;
        public IntPtr Buffer;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct RTL_USER_PROCESS_PARAMETERS
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
        public byte[] Reserved1;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 10)]
        public IntPtr[] Reserved2;

        public UNICODE_STRING ImagePathName;
        public UNICODE_STRING CommandLine;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PEB
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public byte[] Reserved1;

        public byte BeingDebugged;
        public byte Reserved2;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public IntPtr[] Reserved3;

        public IntPtr Ldr;
        public IntPtr ProcessParameters;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct PROCESS_BASIC_INFORMATION
    {
        public IntPtr Reserved1;
        public IntPtr PebBaseAddress;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 2)]
        public IntPtr[] Reserved2;

        public IntPtr UniqueProcessId;
        public IntPtr InheritedFromUniqueProcessId;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenProcess(int dwDesiredAccess, bool bInheritHandle, int dwProcessId);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern bool QueryFullProcessImageName(
        IntPtr hProcess,
        int dwFlags,
        StringBuilder lpExeName,
        ref int lpdwSize);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool ReadProcessMemory(
        IntPtr hProcess,
        IntPtr lpBaseAddress,
        byte[] lpBuffer,
        int dwSize,
        out IntPtr lpNumberOfBytesRead);

    [DllImport("ntdll.dll")]
    private static extern int NtQueryInformationProcess(
        IntPtr processHandle,
        int processInformationClass,
        ref PROCESS_BASIC_INFORMATION processInformation,
        int processInformationLength,
        out int returnLength);

    private static int Main(string[] args)
    {
        int pid;
        List<string> fields;
        if (!TryParseRequest(args, out pid, out fields))
        {
            return 1;
        }

        Dictionary<string, string> values;
        try
        {
            values = QueryProcessValues(pid, fields);
        }
        catch
        {
            values = BuildEmptyValues(fields);
        }

        foreach (KeyValuePair<string, string> pair in values)
        {
            Console.WriteLine(pair.Key + "=" + pair.Value);
        }

        Console.WriteLine();
        return 0;
    }

    private static bool TryParseRequest(string[] args, out int pid, out List<string> fields)
    {
        string joined = string.Join(" ", args);
        Match match = Regex.Match(joined, @"processid\s*=\s*'?(?<pid>\d+)'?", RegexOptions.IgnoreCase);
        pid = match.Success ? int.Parse(match.Groups["pid"].Value) : 0;

        fields = new List<string>();
        int getIndex = Array.FindIndex(args, arg => string.Equals(arg, "get", StringComparison.OrdinalIgnoreCase));
        if (getIndex >= 0)
        {
            for (int i = getIndex + 1; i < args.Length; i++)
            {
                if (args[i].StartsWith("/"))
                {
                    break;
                }

                foreach (string field in args[i].Split(','))
                {
                    string cleaned = field.Trim();
                    if (cleaned.Length > 0)
                    {
                        fields.Add(cleaned.ToLowerInvariant());
                    }
                }
            }
        }

        if (fields.Count == 0)
        {
            fields.Add("commandline");
        }

        return pid > 0;
    }

    private static Dictionary<string, string> QueryProcessValues(int pid, List<string> fields)
    {
        IntPtr handle = OpenProcess(
            PROCESS_QUERY_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ,
            false,
            pid);

        if (handle == IntPtr.Zero)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        try
        {
            PROCESS_BASIC_INFORMATION basicInfo = QueryBasicInformation(handle);
            Dictionary<string, string> values = new Dictionary<string, string>();

            foreach (string field in fields)
            {
                switch (field)
                {
                    case "commandline":
                        values["CommandLine"] = QueryCommandLine(handle, basicInfo);
                        break;
                    case "executablepath":
                        values["ExecutablePath"] = QueryExecutablePath(handle);
                        break;
                    case "parentprocessid":
                        values["ParentProcessId"] = basicInfo.InheritedFromUniqueProcessId.ToInt64().ToString();
                        break;
                    case "processid":
                        values["ProcessId"] = pid.ToString();
                        break;
                    default:
                        values[field] = string.Empty;
                        break;
                }
            }

            return values;
        }
        finally
        {
            CloseHandle(handle);
        }
    }

    private static Dictionary<string, string> BuildEmptyValues(List<string> fields)
    {
        Dictionary<string, string> values = new Dictionary<string, string>();
        foreach (string field in fields)
        {
            switch (field)
            {
                case "commandline":
                    values["CommandLine"] = string.Empty;
                    break;
                case "executablepath":
                    values["ExecutablePath"] = string.Empty;
                    break;
                case "parentprocessid":
                    values["ParentProcessId"] = string.Empty;
                    break;
                case "processid":
                    values["ProcessId"] = string.Empty;
                    break;
                default:
                    values[field] = string.Empty;
                    break;
            }
        }

        return values;
    }

    private static PROCESS_BASIC_INFORMATION QueryBasicInformation(IntPtr handle)
    {
        PROCESS_BASIC_INFORMATION info = new PROCESS_BASIC_INFORMATION();
        int returnLength;
        int status = NtQueryInformationProcess(
            handle,
            ProcessBasicInformation,
            ref info,
            Marshal.SizeOf(typeof(PROCESS_BASIC_INFORMATION)),
            out returnLength);

        if (status != 0)
        {
            throw new InvalidOperationException("NtQueryInformationProcess failed: " + status);
        }

        return info;
    }

    private static T ReadRemoteStruct<T>(IntPtr handle, IntPtr address) where T : struct
    {
        int size = Marshal.SizeOf(typeof(T));
        byte[] buffer = new byte[size];
        IntPtr bytesRead;
        bool ok = ReadProcessMemory(handle, address, buffer, size, out bytesRead);
        if (!ok)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        GCHandle pinned = GCHandle.Alloc(buffer, GCHandleType.Pinned);
        try
        {
            return (T)Marshal.PtrToStructure(pinned.AddrOfPinnedObject(), typeof(T));
        }
        finally
        {
            pinned.Free();
        }
    }

    private static string QueryCommandLine(IntPtr handle, PROCESS_BASIC_INFORMATION basicInfo)
    {
        PEB peb = ReadRemoteStruct<PEB>(handle, basicInfo.PebBaseAddress);
        RTL_USER_PROCESS_PARAMETERS parameters =
            ReadRemoteStruct<RTL_USER_PROCESS_PARAMETERS>(handle, peb.ProcessParameters);

        if (parameters.CommandLine.Buffer == IntPtr.Zero || parameters.CommandLine.Length == 0)
        {
            return string.Empty;
        }

        byte[] buffer = new byte[parameters.CommandLine.Length];
        IntPtr bytesRead;
        bool ok = ReadProcessMemory(
            handle,
            parameters.CommandLine.Buffer,
            buffer,
            buffer.Length,
            out bytesRead);

        if (!ok)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        return Encoding.Unicode.GetString(buffer);
    }

    private static string QueryExecutablePath(IntPtr handle)
    {
        int size = 32768;
        StringBuilder builder = new StringBuilder(size);
        bool ok = QueryFullProcessImageName(handle, 0, builder, ref size);
        if (!ok)
        {
            throw new Win32Exception(Marshal.GetLastWin32Error());
        }

        return builder.ToString();
    }
}
