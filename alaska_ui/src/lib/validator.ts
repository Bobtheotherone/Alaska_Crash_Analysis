import { DataPrepState, ValidationResults, ColumnStat } from '../App';

const BASE_UNKNOWN_STRINGS: Set<string> = new Set([
    "no data", "missing value", "null value", "missing", "na", "n/a", "n.a.",
    "none", "null", "nan", "unknown", "unspecified", "not specified",
    "not applicable", "tbd", "tba", "to be determined", "-", "--", "(blank)",
    "blank", "(null)", "?", "prefer not to say", "refused"
]);

const GENERIC_UNKNOWN_SUBSTRINGS: Set<string> = new Set([
    "unknown", "missing", "unspecified", "not specified", "not applicable",
    "n/a", "na", "null", "blank", "tbd", "tba", "to be determined",
    "refused", "prefer not to say", "no data", "no value",
]);

const YES_SET: Set<string> = new Set(["yes", "y", "true", "t"]);
const NO_SET: Set<string> = new Set(["no", "n", "false", "f"]);

const discoverUnknownPlaceholders = (
    data: Record<string, string>[],
    extraUnknowns: string[] = [],
): Set<string> => {
    const discovered: Map<string, number> = new Map();
    const patterns = Array.from(GENERIC_UNKNOWN_SUBSTRINGS).map(part =>
        new RegExp(`\\b${part.replace(/[-\/\\^$*+?.()|[\]{}]/g, '\\$&')}\\b`, 'i')
    );
    const columns = data.length > 0 ? Object.keys(data[0]) : [];
    const normalizedExtras = extraUnknowns.map(s => s.trim().toLowerCase()).filter(s => !!s);
    const seedUnknowns = new Set<string>([
        ...BASE_UNKNOWN_STRINGS,
        ...normalizedExtras,
    ]);

    for (const col of columns) {
        for (const row of data) {
            const value = row[col];
            if (value === null || value === undefined) continue;

            const v = String(value).trim().toLowerCase();
            if (!v || v.length > 80 || seedUnknowns.has(v)) continue;

            if (patterns.some(p => p.test(v))) {
                discovered.set(v, (discovered.get(v) || 0) + 1);
            }
        }
    }

    const frequentNewUnknowns = new Set<string>();
    for (const [token, count] of discovered.entries()) {
        if (count >= 2) {
            frequentNewUnknowns.add(token);
        }
    }

    return new Set([...seedUnknowns, ...frequentNewUnknowns]);
};

export const runValidationLogic = (data: Record<string, string>[], config: DataPrepState): ValidationResults => {
    const rowCount = data.length;
    if (rowCount === 0) return { rowCount: 0, columnCount: 0, droppedColumnCount: 0, columnStats: [] };

    const columns = Object.keys(data[0]);
    const columnCount = columns.length;
    const augmentedUnknowns = discoverUnknownPlaceholders(
        data,
        config.additionalUnknownTokens || [],
    );

    const columnStats: ColumnStat[] = columns.map(col => {
        let unknownCount = 0;
        let yesCount = 0;
        let noCount = 0;
        const validValuesForYesNo: string[] = [];

        for (const row of data) {
            const value = row[col];
            if (value === null || value === undefined || value === '') {
                unknownCount++;
                continue;
            }
            const normalizedValue = String(value).trim().toLowerCase();
            if (augmentedUnknowns.has(normalizedValue)) {
                unknownCount++;
            } else {
                validValuesForYesNo.push(normalizedValue);
            }
        }
        
        for (const val of validValuesForYesNo) {
            if (YES_SET.has(val)) yesCount++;
            else if (NO_SET.has(val)) noCount++;
        }

        const unknownPercent = (unknownCount / rowCount) * 100;
        const totalYesNo = yesCount + noCount;
        const coveragePercent = (totalYesNo / rowCount) * 100;

        let yesNoStats: ColumnStat['yesNoStats'] = null;
        if (totalYesNo > 0) {
            yesNoStats = {
                yesPercent: (yesCount / totalYesNo) * 100,
                noPercent: (noCount / totalYesNo) * 100,
                totalYesNo,
                coveragePercent,
            };
        }

        let status: 'Keep' | 'Drop' = 'Keep';
        let reason: string | null = null;

        if (unknownPercent > config.unknownThreshold) {
            status = 'Drop';
            reason = `Exceeds unknown threshold (${unknownPercent.toFixed(1)}% > ${config.unknownThreshold}%)`;
        } else if (yesNoStats && coveragePercent >= 50.0) {
            if (yesNoStats.yesPercent < config.yesNoThreshold || yesNoStats.noPercent < config.yesNoThreshold) {
                status = 'Drop';
                reason = `Extreme Yes/No imbalance (Yes: ${yesNoStats.yesPercent.toFixed(1)}%, No: ${yesNoStats.noPercent.toFixed(1)}%)`;
            }
        }

        return {
            column: col,
            unknownPercent,
            yesNoStats,
            status,
            reason,
        };
    });

    const droppedColumnCount = columnStats.filter(cs => cs.status === 'Drop').length;

    return {
        rowCount,
        columnCount,
        droppedColumnCount,
        columnStats,
    };
};
