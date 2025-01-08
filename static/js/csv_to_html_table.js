var CsvToHtmlTable = CsvToHtmlTable || {};

CsvToHtmlTable = {
    init: function (options) {
        options = options || {};
        var csv_path = options.csv_path || "";
        var el = options.element || "table-container";
        var allow_download = options.allow_download || false;
        var csv_options = options.csv_options || {};
        var datatables_options = options.datatables_options || {};
        var custom_formatting = options.custom_formatting || [];
        var customTemplates = {};
        $.each(custom_formatting, function (i, v) {
            var colIdx = v[0];
            var func = v[1];
            customTemplates[colIdx] = func;
        });

        var $table = $("<table class='table table-striped table-condensed' id='" + el + "-table'></table>");
        var $containerElement = $("#" + el);
        $containerElement.empty().append($table);

        $.when($.get(csv_path)).then(
            function (data) {
                var csvData = $.csv.toArrays(data, csv_options);
                var $tableHead = $("<thead></thead>");
                var csvHeaderRow = csvData[0];

                var $tableHeadRow1 = $("<tr></tr>");
                $tableHeadRow1.append($("<th colspan='3' data-dt-order='disable'></th>").text("Metadata"));
                $tableHeadRow1.append($("<th colspan='5'></th>").text("Reasoning Questions"));
                $tableHeadRow1.append($("<th colspan='6'></th>").text("Descriptive Questions"));
                $tableHeadRow1.css("background-color", "#f5f5f5");
                $tableHead.append($tableHeadRow1);

                var $tableHeadRow2 = $("<tr></tr>");
                const explanations = ["The name of the model", 
                "The weight availability of the model", 
                "Vision and language module size in terms of billion parameters",
                "Overall score for reasoning questions", 
                "Score for text-in-chart answers", 
                "Score for text-in-general answers", 
                "Score for number-in-chart answers", 
                "Score for number-in-general answers", 
                "Overall score for descriptive questions", 
                "Score for information extraction questions", 
                "Score for enumeration questions", 
                "Score for pattern recognition questions", 
                "Score for counting questions", 
                "Score for compositionality questions"];
                for (var headerIdx = 0; headerIdx < csvHeaderRow.length; headerIdx++) {
                    var explanation = explanations[headerIdx];
                    $tableHeadRow2Cell = $("<th class='tooltip'></th>").text(csvHeaderRow[headerIdx]);
                    $tableHeadRow2Cell.append($("<span class='tooltiptext'></span>").text(explanation));
                    $tableHeadRow2.append($tableHeadRow2Cell);
                }
                $tableHeadRow2.css("background-color", "#f5f5f5");
                $tableHead.append($tableHeadRow2);

                $table.append($tableHead);
                var $tableBody = $("<tbody></tbody>");

                for (var rowIdx = 1; rowIdx < csvData.length; rowIdx++) {
                    var $tableBodyRow = $("<tr></tr>");
                    for (var colIdx = 0; colIdx < csvData[rowIdx].length; colIdx++) {
                        var $tableBodyRowTd = $("<td></td>");
                        var cellTemplateFunc = customTemplates[colIdx];
                        if (cellTemplateFunc) {
                            $tableBodyRowTd.html(cellTemplateFunc(csvData[rowIdx][colIdx]));
                        } else {
                            $tableBodyRowTd.text(csvData[rowIdx][colIdx]);
                        }
                        if (colIdx == 0) {
                            // left align
                            $tableBodyRowTd.css("text-align", "left");
                            // click to see more info
                            var cellValue = csvData[rowIdx][colIdx];
                            var url = './data/sources/' + cellValue + '.txt';
                            $tableBodyRowTd.html('<a href="' + url + '">' + cellValue + '</a>');
                        }
                        if (colIdx == 2 || colIdx == 7) {
                            $tableBodyRowTd.css("border-right", "1px solid #dbdbdb");
                        }
                        // if the second column equals to "Proprietary", then set the background color of the row to light red
                        if (colIdx == 1 && csvData[rowIdx][colIdx] == "Proprietary") {
                            $tableBodyRow.css("background-color", "#FEFAE3");
                        }
                        // if the second column equals to "Open", then set the background color of the row to light green
                        if (colIdx == 1 && csvData[rowIdx][colIdx] == "Open") {
                            $tableBodyRow.css("background-color", "#F8FBFD");
                        }
                        // if the second column equals to "Domain-specific", then set the background color of the row to light yellow
                        if (colIdx == 1 && csvData[rowIdx][colIdx] == "Domain-specific") {
                            $tableBodyRow.css("background-color", "#ECFFE6");
                        }
                        // if N/A, light blue
                        if (csvData[rowIdx][colIdx] == "N/A") {
                            $tableBodyRow.css("background-color", "#FEF4E4");
                        }


                        $tableBodyRow.append($tableBodyRowTd);
                        $tableBody.append($tableBodyRow);
                    }
                }
                $table.append($tableBody);
                $table.DataTable(datatables_options);
                if (allow_download) {
                    $containerElement.append("<p><a class='btn btn-info' href='" + csv_path + "'><i class='glyphicon glyphicon-download'></i> Download as CSV</a></p>");
                }
            });
    }
};