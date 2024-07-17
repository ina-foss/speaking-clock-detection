# Speaking Clock Detection

This tool can be used to detect on which channel a [speaking clock](https://en.wikipedia.org/wiki/Speaking_clock) is present. It only works on stereo audio files, where either the left or right channel contain only the speaking clock. This tool has only been tested with the French official Speaking Clock.

## Installation

```bash
pip3 install speaking_clock_detection
```

## Usage

The help is available with the following command:
```bash
speaking_clock_detection --help
```

The tool can be used like this:
```bash
speaking_clock_detection \
	--media /file/to/detect/speaking_clock.wav
```

It will output one of the three following values:
- `SPEAKING_CLOCK_TRACK` followed by the channel track id (typically 0 or 1)
- `SPEAKING_CLOCK_NONE` if no speaking clock has been detected
- `SPEAKING_CLOCK_MULTIPLE` if multiple speaking clocks have been detected (this is usally an error)

