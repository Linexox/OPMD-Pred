package org.opmd.followup.demo;

/** Heuristic display hints only. No lesion, site, sharpness or diagnostic assessment. */
final class CaptureHints {
    private int[] previous;
    private double previousMean;
    private String candidate = "";
    private int count;
    private String shown = "请对准要记录的位置";

    void reset() { previous = null; candidate = ""; count = 0; shown = "请对准要记录的位置"; }

    String update(int[] samples) {
        double mean = 0;
        int dark = 0, bright = 0;
        for (int value : samples) { mean += value; if (value < 20) dark++; if (value > 245) bright++; }
        mean /= samples.length;
        double change = 0;
        if (previous != null && previous.length == samples.length) {
            for (int i = 0; i < samples.length; i++)
                change += Math.abs((samples[i] - mean) - (previous[i] - previousMean));
            change /= samples.length;
        }
        String next = (double) dark / samples.length > .4 ? "画面暗部较多，请调整光线"
            : (double) bright / samples.length > .25 ? "画面亮部较多，请调整角度或光线"
            : previous != null && change > 18 ? "画面变化较大，请稳住手机"
            : "保持手机平稳，检查取景后拍摄";
        previous = samples.clone(); previousMean = mean;
        if (candidate.equals(next)) count++; else { candidate = next; count = 1; }
        if (count >= 3) shown = next;
        return shown;
    }
}
