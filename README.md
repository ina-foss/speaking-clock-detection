# Speaking Clock Detection

This tool can be used to detect on which channel a [speaking clock](https://en.wikipedia.org/wiki/Speaking_clock) is present. It only works on stereo audio files, where either the left or right channel contain only the speaking clock. This tool has only been tested with the French official Speaking Clock.

## Installation

```bash
pip3 install speaking-clock-detection
```

## Usage

The help is available with the following command:
```bash
speaking_clock_detection --help
```

```
options:
  -h, --help            show this help message and exit
  -m MEDIA, --media MEDIA
                        full path to media to analyze
  -t TMPDIR, --tmpdir TMPDIR
                        Temporary directory used to store intermediate files. Should be a fast access
                        directory such as Ram Disk or SSD hard drive. Default value: /dev/shm (linux ram
                        disk)
  -o OUTPUT, --output OUTPUT
                        output file for the result. Default value: /dev/stdout.
  -f FFMPEG, --ffmpeg FFMPEG
                        Full path to ffmpeg binary. If not provided, this will used default binary
                        installed on the system. This program has been tested with ffmpeg version
                        2.8.8-0ubuntu0.16.04.1
```

### Example
The tool can be used like this:
```bash
speaking_clock_detection \
	--media /file/to/detect/speaking_clock.wav
```

It will output one of the three following values:
- `SPEAKING_CLOCK_TRACK` followed by the channel track id (typically 0 or 1)
- `SPEAKING_CLOCK_NONE` if no speaking clock has been detected
- `SPEAKING_CLOCK_MULTIPLE` if multiple speaking clocks have been detected (this is usually an error)

